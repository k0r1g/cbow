"""
02_train_reg.py – predict Hacker News up‑votes from titles
---------------------------------------------------------
• Re‑query the database so we can pull *timestamps* in addition to scores.
• Down‑sample zero‑up‑vote rows so positives == negatives.
• Tokenise each title with the pre‑built lookup tables, pad within batch.
• Embed with the fine‑tuned Word2Vec → mean‑pool → concatenate extra features:
      – title length (tokens)
      – month‑level mean score (provides temporal trend signal)
• Train an MLP (`model.Regressor`) using **Huber loss** (SmoothL1Loss).
• Train/val split is 80/20 and results logged to wandb.
"""

from __future__ import annotations

import os
import datetime
import random
from pathlib import Path
from collections import Counter, defaultdict
import re  # Added import

import pandas as pd
import numpy as np
import torch
import tqdm
import wandb
import psycopg2

# Import NLTK dependencies for preprocessing
try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
except ImportError:
    import os
    print("NLTK not found, installing...")
    os.system("pip install -q nltk")
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import nltk
    print("Downloading NLTK data (punkt, stopwords)...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

import model          # CBOW + Regressor
import evaluate        # neighbour checks (optional)

# ---------------------------------------------------------------------------
# Config – paths & hyper‑params
# ---------------------------------------------------------------------------
DB_URL = "postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
CBOW_CKPT = "checkpoints/latest_hn_w2v.word2vec"   # ← set to your checkpoint
PAD_IDX = 0
EPOCHS = 5
BATCH_SIZE = 512
LR = 3e-4  # Fixed the dash character
HUBER_DELTA = 10.0
SEED = 42

# ---------------------------------------------------------------------------
# 1️⃣  Fetch data (title, score, time) & build DataFrame
# ---------------------------------------------------------------------------
print("Connecting to Postgres …")
conn = psycopg2.connect(DB_URL)
query = (
    "SELECT title, score, time \n"
    "  FROM \"hacker_news\".\"items\" \n"
    " WHERE title IS NOT NULL AND score IS NOT NULL AND time IS NOT NULL"
)
items = pd.read_sql_query(query, conn)
conn.close()
print(f"Fetched {len(items):,} rows from DB")

# Convert epoch → datetime → year‑month bucket
items["dt"] = pd.to_datetime(items["time"], unit="s", utc=True)
items["ym"] = items["dt"].dt.to_period("M").astype(str)

# ---------------------------------------------------------------------------
# 2️⃣  Down‑sample zero‑score rows so |score==0| == |score>0|
# ---------------------------------------------------------------------------
zeros = items[items.score == 0]
positives = items[items.score > 0]
zeros_sampled = zeros.sample(n=len(positives), random_state=42)
balanced = pd.concat([positives, zeros_sampled]).sample(frac=1, random_state=42)
print("Balanced set:", Counter(balanced.score == 0))

# ---------------------------------------------------------------------------
# 3️⃣  Build month‑level mean scores (temporal feature)
# ---------------------------------------------------------------------------
month_mean = balanced.groupby("ym").score.mean().to_dict()
balanced["month_avg"] = balanced["ym"].map(month_mean)

# ---------------------------------------------------------------------------
# Shared Preprocessing Function (from 00_train_tkn.py)
# ---------------------------------------------------------------------------
def preprocess(text: str) -> list[str]:
  """Lowercase, remove punctuation/hyphens, tokenize, remove stopwords."""
  if not isinstance(text, str):
    return []
    
  text = text.lower()
  # Remove punctuation but keep important word boundaries
  text = re.sub(r'[^\w\s-]', ' ', text)
  # Replace hyphens with space to separate hyphenated words
  text = text.replace('-', ' ')
  
  # Tokenize using NLTK
  words = word_tokenize(text)
  
  # Remove stopwords
  stop_words = set(stopwords.words('english'))
  words = [word for word in words if word not in stop_words and word.isalpha()] # Added isalpha check
  
  # Note: Frequency filtering is omitted here as we assume the vocab from 
  # words_to_ids.pkl already handles this implicitly.
  
  return words

# ---------------------------------------------------------------------------
# 4️⃣  Tokenise titles with existing vocab → token‑id list
# ---------------------------------------------------------------------------
print("Loading word to ID mapping...")
words_to_ids_path = Path("words_to_ids.pkl")
if not words_to_ids_path.exists():
    raise FileNotFoundError(
        f"Error: {words_to_ids_path} not found. "
        "Please run 00_train_tkn.py first to generate token mappings."
    )
words_to_ids = __import__("pickle").load(open(words_to_ids_path, "rb"))

stop_token = PAD_IDX

def tokens_to_ids(title: str):
    # Use the shared preprocess function
    processed_tokens = preprocess(title)
    # Map tokens to IDs, using PAD_IDX for unknown words
    ids = [words_to_ids.get(tok, stop_token) for tok in processed_tokens]
    # Ensure we always return a list, even if empty after preprocessing
    return ids if ids else [stop_token]

print("Tokenizing titles using preprocess function...")
balanced["ids"] = balanced.title.apply(tokens_to_ids)

# Display a sample of original vs tokenized
print("\nSample tokenization:")
sample_idx = balanced.sample(1).index[0]
print("Original Title:", balanced.loc[sample_idx, 'title'])
print("Token IDs:", balanced.loc[sample_idx, 'ids'])

# ---------------------------------------------------------------------------
# 5️⃣  Train/val/test split (80/10/10)
# ---------------------------------------------------------------------------
# First, split into 80% train and 20% temporary (for val + test)
train_df = balanced.sample(frac=0.8, random_state=SEED)
temp_df = balanced.drop(train_df.index)

# Now split the temporary 20% into 10% validation and 10% test (50% of temp_df)
val_df = temp_df.sample(frac=0.5, random_state=SEED)
test_df = temp_df.drop(val_df.index)

print("\nData Split:")
print(f"  Train samples: {len(train_df):,}")
print(f"  Validation samples: {len(val_df):,}")
print(f"  Test samples: {len(test_df):,}")

# ---------------------------------------------------------------------------
# 6️⃣  DataLoader helpers
# ---------------------------------------------------------------------------

def pad_collate(batch):
    ids, scores, month_avg, length = zip(*batch)
    max_len = max(len(seq) for seq in ids)
    padded = [torch.tensor(seq + [PAD_IDX] * (max_len - len(seq))) for seq in ids]
    tokens = torch.stack(padded)
    extra = torch.stack(
        [
            torch.tensor([la, ma], dtype=torch.float32)
            for la, ma in zip(length, month_avg)
        ]
    )
    scores = torch.tensor(scores, dtype=torch.float32).unsqueeze(1)
    return tokens, extra, scores

class HNRegDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.ids = df["ids"].tolist()
        self.scores = df.score.tolist()
        self.month_avg = df.month_avg.tolist()
        self.lengths = [len(seq) for seq in self.ids]

    def __len__(self):  # type: ignore[override]
        return len(self.ids)

    def __getitem__(self, idx):  # type: ignore[override]
        return (
            self.ids[idx],
            self.scores[idx],
            self.month_avg[idx],
            self.lengths[idx],
        )

train_dl = torch.utils.data.DataLoader(
    HNRegDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate
)
val_dl = torch.utils.data.DataLoader(
    HNRegDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate
)
# Create DataLoader for the test set
test_dl = torch.utils.data.DataLoader(
    HNRegDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate
)

# ---------------------------------------------------------------------------
# 7️⃣  Models & optimiser
# ---------------------------------------------------------------------------
cbow = model.CBOW(CBOW_CKPT, trainable=False).eval()
emb_dim = cbow.emb.embedding_dim
extra_dim = 2  # length + month_avg
reg = model.Regressor(emb_dim + extra_dim)

opt = torch.optim.Adam(reg.parameters(), lr=LR)
criterion = torch.nn.SmoothL1Loss(beta=HUBER_DELTA)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cbow, reg = cbow.to(device), reg.to(device)

# ---------------------------------------------------------------------------
# 8️⃣  wandb setup
# ---------------------------------------------------------------------------
wandb.init(
    project="mlx7‑week1‑hn‑regression",
    name=f"{datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.reg",
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "loss": "Huber",
        "extra_features": ["length", "month_avg"],
    },
)

# ---------------------------------------------------------------------------
# 9️⃣  Training loop
# ---------------------------------------------------------------------------
for epoch in range(EPOCHS):
    reg.train()
    pbar = tqdm.tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for tokens, extra, y in pbar:
        tokens, extra, y = tokens.to(device), extra.to(device), y.to(device)
        with torch.no_grad():
            emb = cbow.emb(tokens)              # (B,T,D)
            mask = (tokens != PAD_IDX).unsqueeze(-1)
            summed = (emb * mask).sum(dim=1)
            lengths = mask.sum(dim=1) + 1e-9
            mean_emb = summed / lengths         # (B,D)
        x = torch.cat([mean_emb, extra], dim=1)
        y_hat = reg(x)
        loss = criterion(y_hat, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        wandb.log({"train_loss": loss.item()})

    # ------ validation ------
    reg.eval()
    losses = []
    with torch.no_grad():
        for tokens, extra, y in val_dl:
            tokens, extra, y = tokens.to(device), extra.to(device), y.to(device)
            emb = cbow.emb(tokens)
            mask = (tokens != PAD_IDX).unsqueeze(-1)
            mean_emb = (emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
            x = torch.cat([mean_emb, extra], dim=1)
            y_hat = reg(x)
            loss = criterion(y_hat, y)
            losses.append(loss.item())
    val_loss = float(np.mean(losses))
    print(f"Epoch {epoch+1} – val_loss {val_loss:.4f}")
    wandb.log({"val_loss": val_loss})

# ---------------------------------------------------------------------------
# 🔚  Save final regressor
# ---------------------------------------------------------------------------
timestamp = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
ckpt_out = f"checkpoints/{timestamp}.reg.pth"
Path("checkpoints").mkdir(exist_ok=True)
torch.save(reg.state_dict(), ckpt_out)
print("Saved final regressor to", ckpt_out)
wip_ckpt_path = Path(ckpt_out) # Keep track of the saved path
wip_cbow_ckpt = CBOW_CKPT # Keep track of the CBOW model used
wandb.finish()

# ---------------------------------------------------------------------------
# 🔟 Final Evaluation on Test Set
# ---------------------------------------------------------------------------
print("\nStarting final evaluation on the test set...")

# Load the saved model state (ensure we test the saved version)
final_regressor = model.Regressor(emb_dim + extra_dim)
final_regressor.load_state_dict(torch.load(wip_ckpt_path))
final_regressor = final_regressor.to(device)
final_regressor.eval() # Set to evaluation mode

# Load the CBOW model used during training (ensure consistency)
# Use wip_cbow_ckpt which was determined before training loop
final_cbow = model.CBOW(wip_cbow_ckpt, trainable=False).eval().to(device)

test_losses = []
with torch.no_grad():
    pbar_test = tqdm.tqdm(test_dl, desc="Testing")
    for tokens, extra, y in pbar_test:
        tokens, extra, y = tokens.to(device), extra.to(device), y.to(device)
        
        # Get embeddings from the consistent CBOW model
        emb = final_cbow.emb(tokens)
        mask = (tokens != PAD_IDX).unsqueeze(-1)
        mean_emb = (emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        
        # Combine features and predict
        x = torch.cat([mean_emb, extra], dim=1)
        y_hat = final_regressor(x)
        
        # Calculate loss
        loss = criterion(y_hat, y)
        test_losses.append(loss.item())
        pbar_test.set_postfix({"batch_loss": f"{loss.item():.4f}"})

final_test_loss = float(np.mean(test_losses))
print(f"\n---")
print(f"✅ Final Test Loss: {final_test_loss:.4f}")
print(f"---")

# ---------------------------------------------------------------------------
# Upload the final model to HuggingFace
# ---------------------------------------------------------------------------
print("\nUploading regressor model to Hugging Face Hub...")

try:
    # Import and install huggingface_hub if needed
    try:
        from huggingface_hub import login, HfApi
    except ImportError:
        import os
        print("Installing huggingface_hub...")
        os.system("pip install -q huggingface_hub")
        from huggingface_hub import login, HfApi
    
    # Log in to Hugging Face
    print("Logging in to Hugging Face Hub...")
    login()
    
    # Initialize API
    api = HfApi()
    
    # Create a new model repository
    repo_id = "Kogero/hackernews-score-regressor"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    
    # Create a dictionary with metadata to save alongside the model
    model_info = {
        "embedding_dimension": emb_dim,
        "extra_features_dimension": extra_dim,
        "total_input_dimension": emb_dim + extra_dim,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "huber_delta": HUBER_DELTA,
        "validation_loss": val_loss,
        "word2vec_checkpoint": CBOW_CKPT,
        "training_samples": len(train_df),
        "validation_samples": len(val_df),
        "timestamp": timestamp
    }
    
    # Save the metadata to a JSON file
    import json
    with open("regressor_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    # Create a model card with information
    model_card = f"""---
language:
- en
license: mit
tags:
- hacker-news
- regression
- score-prediction
- word2vec
---

# Hacker News Score Regressor

This repository contains a regression model trained to predict the score (upvotes) of Hacker News posts based on their titles.

## Model Details

- **Type**: MLP Regression model
- **Input**: 
  - Word embeddings from a fine-tuned Word2Vec model (dimension: {emb_dim})
  - Extra features: title length and month-level average score (dimension: {extra_dim})
- **Total input dimension**: {emb_dim + extra_dim}
- **Loss function**: Huber Loss (SmoothL1Loss) with beta={HUBER_DELTA}
- **Final validation loss**: {val_loss:.4f}

## Training Details

- **Word embeddings**: Used pre-trained Word2Vec model fine-tuned on Hacker News titles
- **Training samples**: {len(train_df):,}
- **Validation samples**: {len(val_df):,}
- **Epochs**: {EPOCHS}
- **Batch size**: {BATCH_SIZE}
- **Learning rate**: {LR}
- **Data balancing**: Equal number of zero-score and positive-score posts

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from model import CBOW, Regressor  # Assuming you have the same model definitions

# Download the model
regressor_file = hf_hub_download(repo_id="Kogero/hackernews-score-regressor", filename="model.pth")
metadata_file = hf_hub_download(repo_id="Kogero/hackernews-score-regressor", filename="regressor_info.json")

# Load metadata
import json
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

# Create and load the regressor
regressor = Regressor(metadata['total_input_dimension'])
regressor.load_state_dict(torch.load(regressor_file))
regressor.eval()

# To use it, you'll need:
# 1. A tokenized title with token IDs
# 2. The CBOW model for embeddings
# 3. Extra features (title length and month average)
```

## Pipeline Description

1. Titles are tokenized using the same vocabulary as the Word2Vec model
2. Tokens are converted to embeddings using the CBOW model
3. Embeddings are mean-pooled to create a fixed-length representation
4. Extra features (title length and month-level average score) are concatenated
5. The combined vector is passed through the regressor MLP to predict the score

## Performance

The model was trained to minimize Huber loss on a balanced dataset of Hacker News posts.
"""
    
    # Save the model card to a file
    with open("README.md", "w") as f:
        f.write(model_card)
    
    # Upload the files
    print("Uploading files to Hugging Face Hub...")
    
    # Upload the model
    api.upload_file(
        path_or_fileobj=ckpt_out,
        path_in_repo="model.pth",
        repo_id=repo_id,
        repo_type="model"
    )
    
    # Upload metadata
    api.upload_file(
        path_or_fileobj="regressor_info.json",
        path_in_repo="regressor_info.json",
        repo_id=repo_id,
        repo_type="model"
    )
    
    # Upload README
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )
    
    # Create and upload sample inference code
    sample_code = """import torch
import json
from huggingface_hub import hf_hub_download
from model import CBOW, Regressor  # Import your model definitions

def load_models():
    # Download files from Hugging Face
    regressor_path = hf_hub_download(repo_id="Kogero/hackernews-score-regressor", 
                                     filename="model.pth")
    metadata_path = hf_hub_download(repo_id="Kogero/hackernews-score-regressor", 
                                    filename="regressor_info.json")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load word2vec model (you'll need to download this separately)
    cbow = CBOW(metadata['word2vec_checkpoint'], trainable=False)
    
    # Create and load regressor
    regressor = Regressor(metadata['total_input_dimension'])
    regressor.load_state_dict(torch.load(regressor_path))
    regressor.eval()
    
    return cbow, regressor, metadata

def predict_score(title, cbow, regressor, month_avg=None):
    # Tokenize title (simplified)
    tokens = title.lower().split()
    
    # Convert to token IDs (you'll need to use your tokenizer)
    # This is just a placeholder
    token_ids = [1, 2, 3]  # Replace with actual tokenization
    
    # Get features
    title_length = len(token_ids)
    if month_avg is None:
        # Use a reasonable default
        month_avg = 5.0
    
    # Convert to tensors
    tokens_tensor = torch.tensor([token_ids])
    
    # Get embeddings and mean-pool
    with torch.no_grad():
        token_embeddings = cbow.emb(tokens_tensor)
        mean_embedding = token_embeddings.mean(dim=1)
        
        # Combine with extra features
        extra = torch.tensor([[title_length, month_avg]], dtype=torch.float32)
        combined = torch.cat([mean_embedding, extra], dim=1)
        
        # Get prediction
        score = regressor(combined).item()
    
    return score

# Example usage
if __name__ == "__main__":
    cbow, regressor, metadata = load_models()
    title = "Show HN: A new way to think about machine learning"
    predicted_score = predict_score(title, cbow, regressor)
    print(f"Predicted score: {predicted_score:.1f}")
"""
    
    with open("sample_inference.py", "w") as f:
        f.write(sample_code)
    
    api.upload_file(
        path_or_fileobj="sample_inference.py",
        path_in_repo="sample_inference.py",
        repo_id=repo_id,
        repo_type="model"
    )
    
    print(f"\nSuccess! Model uploaded to: https://huggingface.co/{repo_id}")

except Exception as e:
    print(f"Error uploading to Hugging Face Hub: {e}")
    print("You can still use the local model from:", ckpt_out)
