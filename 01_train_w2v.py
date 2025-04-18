# 01_train_w2v.py – fine‑tune Text8 Word2Vec on Hacker News tokens
# -----------------------------------------------------------------
# This is the production training script that ties together:
#   • gensim → fine‑tuned embeddings
#   • model.CBOW – Torch wrapper around the gensim checkpoint (for eval)
#   • evaluate.topk – nearest‑neighbour monitoring each epoch
#   • dataset.HNTitles – (imported for downstream compatibility)
#
# It produces a *.word2vec checkpoint after every epoch and immediately wraps
# it with `model.CBOW` so `evaluate.topk` can inspect the updated embeddings
# using the *same* vocabulary/lookup tables as the rest of the pipeline.
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
import random
import pickle
import datetime
from pathlib import Path

import numpy as np
import tqdm
import wandb

# External deps
from gensim.models import Word2Vec
import gensim.downloader as api

# Local project modules
import model        # provides CBOW & Regressor classes
import evaluate     # unchanged helper from legacy code
import dataset      # HNTitles* datasets (imported to ensure availability)

# ---------------------------------------------------------------------------
# Reproducibility & run‑stamp
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

TS = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
RUN_NAME = f"{TS}.hn‑w2v‑ft"

# ---------------------------------------------------------------------------
# Config – base Text8 model path (None → download), training hyper‑params
# ---------------------------------------------------------------------------
TEXT8_MODEL_PATH: str | None = None   # e.g. 'text8.word2vec'
HN_TOKENS_PATH = "title_tokens.pkl"
EPOCHS = 3
BATCH_SIZE = 10000                   # sentences per mini‑batch (≈ titles)

# ---------------------------------------------------------------------------
# Load Hacker News tokenised titles (list[list[str]])
# ---------------------------------------------------------------------------
with open(HN_TOKENS_PATH, "rb") as f:
    hn_corpus: list[list[str]] = pickle.load(f)

print(f"Loaded Hacker News corpus  –  {len(hn_corpus):,} titles")
print("Sample:", hn_corpus[:3])

# ---------------------------------------------------------------------------
# Load / download pre‑trained Text8 model
# ---------------------------------------------------------------------------
print("\nLoading Text8 Word2Vec base model …")
if TEXT8_MODEL_PATH and os.path.isfile(TEXT8_MODEL_PATH):
    w2v: Word2Vec = Word2Vec.load(TEXT8_MODEL_PATH)
else:
    w2v = api.load("word2vec‑text8")            # 100‑D, CBOW, window=5
print("Model loaded:")
print("  vector_size =", w2v.vector_size)
print("  vocab        =", len(w2v.wv))

# ---------------------------------------------------------------------------
# Extend vocabulary with Hacker News words
# ---------------------------------------------------------------------------
print("\nUpdating vocabulary with Hacker News tokens …")
w2v.build_vocab(hn_corpus, update=True)
print("New vocab size:", len(w2v.wv))

# ---------------------------------------------------------------------------
# Prepare output folder & wandb run
# ---------------------------------------------------------------------------
Path("checkpoints").mkdir(parents=True, exist_ok=True)

wandb.init(
    project="mlx7‑week1‑word2vec‑hn",
    name=RUN_NAME,
    config={
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "vector_size": w2v.vector_size,
        "window": w2v.window,
        "min_count": w2v.min_count,
        "sg": w2v.sg,
    },
)

# ---------------------------------------------------------------------------
# Optionally verify the Dataset class loads – not used in training loop here
# ---------------------------------------------------------------------------
try:
    ds = dataset.HNTitles()
    print(f"Dataset check – {len(ds):,} tokenised titles available.")
except Exception as exc:
    print("⚠️  Could not load dataset (continuing):", exc)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
TOTAL_SENTS = len(hn_corpus)
for epoch in range(EPOCHS):
    random.shuffle(hn_corpus)
    prgs = tqdm.tqdm(
        range(0, TOTAL_SENTS, BATCH_SIZE),
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
        leave=False,
    )
    for start in prgs:
        batch = hn_corpus[start : start + BATCH_SIZE]
        w2v.train(batch, total_examples=len(batch), epochs=1, compute_loss=True)
        loss = w2v.get_latest_training_loss()
        prgs.set_postfix({"loss": f"{loss:.4f}"})
        wandb.log({"loss": loss})

    # 🗄️  checkpoint
    ckpt_path = f"checkpoints/{RUN_NAME}.epoch{epoch + 1}.word2vec"
    w2v.save(ckpt_path)
    art = wandb.Artifact("word2vec‑weights", type="model")
    art.add_file(ckpt_path)
    wandb.log_artifact(art)

    # 🔍  quick neighbour check via Torch wrapper + evaluate.topk
    cbow = model.CBOW(ckpt_path, trainable=False)
    evaluate.topk(cbow)          # default word='computer'

    print(f"Epoch {epoch + 1} complete – loss {loss:.4f}\n")

# ---------------------------------------------------------------------------
# Final neighbours sanity‑check
# ---------------------------------------------------------------------------
for probe in ["ai", "python", "apple", "openai"]:
    cbow = model.CBOW(ckpt_path, trainable=False)
    evaluate.topk(cbow, word=probe)

wandb.finish()
print("\nTraining finished – final model saved in checkpoints/")

# ---------------------------------------------------------------------------
# Upload the final model to HuggingFace
# ---------------------------------------------------------------------------
print("\nUploading model to HuggingFace Hub...")

try:
    # Import and install huggingface_hub if needed
    try:
        from huggingface_hub import login, HfApi
    except ImportError:
        import os
        print("Installing huggingface_hub...")
        os.system("pip install -q huggingface_hub")
        from huggingface_hub import login, HfApi
    
    # Log in to HuggingFace
    print("Logging in to Hugging Face Hub...")
    login()
    
    # Initialize API
    api = HfApi()
    
    # Create a new model repository
    repo_id = "Kogero/hackernews-word2vec"
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    
    # Create a model card with information
    model_card = f"""---
language:
- en
license: mit
tags:
- word2vec
- hacker-news
- embeddings
---

# Hacker News Word2Vec Embeddings

This repository contains a Word2Vec model fine-tuned on Hacker News titles. The model was initialized with the `text8` Word2Vec model and then fine-tuned on Hacker News data.

## Model Details

- **Base Model**: text8 Word2Vec
- **Vector Size**: {w2v.vector_size}
- **Window Size**: {w2v.window}
- **Min Count**: {w2v.min_count}
- **Model Type**: {"Skip-gram" if w2v.sg else "CBOW"}
- **Vocabulary Size**: {len(w2v.wv)}
- **Training Loss**: {loss:.4f}

## Usage

```python
from gensim.models import Word2Vec
from huggingface_hub import hf_hub_download

# Download the model
model_file = hf_hub_download(repo_id="Kogero/hackernews-word2vec", filename="model.word2vec")

# Load the model
w2v = Word2Vec.load(model_file)

# Use the model
similar_words = w2v.wv.most_similar("python", topn=5)
print(similar_words)
```

## Fine-tuning Details

This model was fine-tuned for {EPOCHS} epochs with a batch size of {BATCH_SIZE}.
"""
    
    # Save the model card to a file
    with open("README.md", "w") as f:
        f.write(model_card)
    
    # Upload the model card and the model file
    print("Uploading model and README...")
    
    # Upload README.md
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )
    
    # Upload the final model
    api.upload_file(
        path_or_fileobj=final_model_path,
        path_in_repo="model.word2vec",
        repo_id=repo_id,
        repo_type="model"
    )
    
    print(f"\nSuccess! Model uploaded to: https://huggingface.co/{repo_id}")

except Exception as e:
    print(f"Error uploading to HuggingFace Hub: {e}")
    print("You can still use the local model from:", final_model_path)
