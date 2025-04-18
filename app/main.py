# FastAPI Application

'''
main.py
There should be 4 endpoints in your FastAPI Application

GET: /ping -> str
This should just return an "ok" string and will be used for a healthcheck
GET: /version -> {"version": str}
This will return the current model version
GET: /logs -> {"logs": [str]}
save the logs somewhere and take them from here
logs need to be persistent
logs should have:
latency
version
timestamp
input
prediction
POST: /how_many_upvotes: {"author": str, "title": str, "timestamp": str} --> {"upvotes": number}
This is where you infer from your model, log, and return the prediction
'''

import os
import pickle
import json
from datetime import datetime
from pathlib import Path

# --- FastAPI ---
from fastapi import FastAPI
from pydantic import BaseModel

# --- ML/AI ---
import torch
# Need to ensure NLTK data is available if not using a pre-built container
try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    # Verify data existence (optional, prevents runtime errors)
    try:
        stopwords.words('english')
    except LookupError:
        import nltk
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
    try:
        word_tokenize("test sentence")
    except LookupError:
        import nltk
        print("Downloading NLTK punkt...")
        nltk.download('punkt', quiet=True)
except ImportError:
    print("NLTK not found. Please install it: pip install nltk")
    # Mock functions to avoid hard crash if NLTK isn't installed
    def word_tokenize(x): return x.split()
    def stopwords(): return set()

# --- Local Model Definitions ---
# Assuming model.py is in the same directory or PYTHONPATH
try:
    from model import CBOW, Regressor
except ImportError as e:
    print(f"Error importing model definitions: {e}")
    print("Please ensure model.py is accessible.")
    # Define dummy classes if import fails to allow app to start (but prediction will fail)
    class CBOW(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            # Minimal implementation to avoid attribute errors later
            self.emb = torch.nn.Embedding(1,1)
            self.vocab_to_int = {}
            self.int_to_vocab = []
            self.embedding_dim = 1
    class Regressor(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.fc = torch.nn.Linear(1,1)
        def forward(self, x): return self.fc(x)


# --- Configuration ---
# Model version (consider deriving from REG_CKPT filename)
REG_CKPT_PATH_STR = "checkpoints/2025_04_18__15_24_31.reg.pth" # Default if latest not found
try:
    # Find the latest regressor checkpoint
    CHECKPOINTS_DIR = Path("checkpoints")
    MODEL_PATTERN = "*.reg.pth"
    checkpoint_files = sorted(CHECKPOINTS_DIR.glob(MODEL_PATTERN), key=os.path.getmtime, reverse=True)
    if checkpoint_files:
        REG_CKPT_PATH_STR = str(checkpoint_files[0])
        print(f"Using latest regressor checkpoint: {REG_CKPT_PATH_STR}")
    else:
        print(f"Warning: No checkpoints found matching '{MODEL_PATTERN}'. Using default: {REG_CKPT_PATH_STR}")
except Exception as e:
     print(f"Error finding latest checkpoint: {e}. Using default: {REG_CKPT_PATH_STR}")

# Derive version from checkpoint name if possible
try:
    model_version = Path(REG_CKPT_PATH_STR).stem.split('.')[0] # Extracts timestamp part
except:
    model_version = "0.1.0" # Fallback

# Log path
log_dir_path = Path(os.getenv("LOG_DIR", "/var/log/app")) # Use env var or default
log_path = log_dir_path / f"V-{model_version}.log"

# Checkpoint Paths (ensure these are correct relative to where the app runs)
CBOW_CKPT = Path("checkpoints/2025_04_18__12_41_13.hn‑w2v‑ft.epoch2.word2vec")
REG_CKPT = Path(REG_CKPT_PATH_STR)
VOCAB_PATH = Path("words_to_ids.pkl")
PAD_IDX = 0

# --- Model Loading ---
print("Loading vocabulary mapping...")
try:
    with open(VOCAB_PATH, "rb") as f:
        words_to_ids = pickle.load(f)
except FileNotFoundError:
    print(f"ERROR: Vocabulary file not found at {VOCAB_PATH}")
    words_to_ids = {"<PAD>": 0} # Dummy vocab
    PAD_IDX = 0

print("Loading models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    cbow = CBOW(CBOW_CKPT, trainable=False).eval().to(device)
    # Add +2 for length and month_avg features
    reg = Regressor(cbow.embedding_dim + 2)
    reg.load_state_dict(torch.load(REG_CKPT, map_location=device))
    reg.eval().to(device)
    print("Models loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR loading model checkpoint: {e}")
    print("Prediction endpoint will likely fail.")
    # Use dummy models if loading fails
    cbow = CBOW(CBOW_CKPT, trainable=False).eval().to(device) # Will likely fail if path is wrong
    reg = Regressor(cbow.embedding_dim + 2).eval().to(device) # Adjust dummy size if needed
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    cbow = CBOW(CBOW_CKPT, trainable=False).eval().to(device)
    reg = Regressor(cbow.embedding_dim + 2).eval().to(device)


# --- FastAPI App Initialization ---
app = FastAPI(title="Hacker News Upvote Predictor", version=model_version)

# --- Pydantic Input Schema ---
class Post(BaseModel):
    author: str
    title: str
    timestamp: str  # ISO format or epoch string


# --- Preprocessing & Prediction Logic ---
_stop_words_cache = None
def get_stopwords():
    global _stop_words_cache
    if _stop_words_cache is None:
        try:
            _stop_words_cache = set(stopwords.words("english"))
        except Exception as e:
             print(f"Warning: Could not load NLTK stopwords: {e}")
             _stop_words_cache = set()
    return _stop_words_cache

def preprocess(text: str) -> list[str]:
    import re
    if not isinstance(text, str): return []
    text = text.lower()
    text = re.sub(r"[^\w\s-]", " ", text) # Keep hyphens for now
    text = text.replace("-", " ") # Replace hyphens with space
    try:
        words = word_tokenize(text)
    except Exception as e:
        print(f"Warning: NLTK word_tokenize failed: {e}")
        words = text.split() # Fallback basic split
    stop_words = get_stopwords()
    return [w for w in words if w.isalpha() and w not in stop_words]

def tokens_to_ids(title: str) -> list[int]:
    tokens = preprocess(title)
    # Ensure at least one token (PAD) if preprocessing results in empty list
    return [words_to_ids.get(w, PAD_IDX) for w in tokens] if tokens else [PAD_IDX]

def predict_upvotes(post: Post) -> float:
    """Predicts upvotes based on post title and metadata."""
    ids = tokens_to_ids(post.title)
    length = len(ids)

    # Simple extraction of year-month. Assumes ISO-like format or similar prefix.
    # Robust parsing might be needed for various timestamp formats.
    try:
        dt = datetime.fromisoformat(post.timestamp.replace("Z", "+00:00")) # Handle Z for UTC
        month_avg = dt.month # Example: Use month number (1-12) - Adjust if needed
    except:
         # Fallback if timestamp format is unexpected
         month_avg = 6.0 # Or some other default/average

    # Convert to tensors and move to device
    tokens_tensor = torch.tensor([ids]).to(device)

    # Prepare model inputs
    with torch.no_grad(): # Ensure no gradients are calculated during inference
        mask = (tokens_tensor != PAD_IDX).unsqueeze(-1).float() # Ensure mask is float for division
        # Handle potential division by zero if mask sum is zero (e.g., only PAD tokens)
        mask_sum = mask.sum(1)
        safe_mask_sum = torch.where(mask_sum == 0, torch.tensor(1e-9, device=device), mask_sum)

        emb = cbow.emb(tokens_tensor) # (B, T, D)
        mean_emb = (emb * mask).sum(1) / safe_mask_sum # (B, D)

        # Create extra features tensor and move to device
        extra = torch.tensor([[length, month_avg]], dtype=torch.float32).to(device) # (B, 2)

        # Concatenate and predict
        x = torch.cat([mean_emb, extra], dim=1) # (B, D+2)
        score = reg(x).item()

    # Post-process prediction (e.g., ensure non-negative, round)
    return max(0.0, round(score, 2))


# --- Logging Functions ---
def log_request(log_path: Path, message: dict):
    """Appends a log message (dict) as a JSON line to the log file."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        log_line = json.dumps(message, ensure_ascii=False) # Convert dict to JSON string
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line + "\\n")
    except Exception as e:
        print(f"Error writing to log file {log_path}: {e}") # Log errors to stderr/stdout

def read_logs(log_path: Path) -> dict:
    """Reads log lines from the file."""
    logs_out = []
    if not log_path.exists():
        return {"logs": logs_out}
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    # Optionally parse JSON back, or just return raw lines
                    # logs_out.append(json.loads(line.strip()))
                    logs_out.append(line.strip()) # Return as raw strings
                except json.JSONDecodeError:
                    logs_out.append(f"Error decoding: {line.strip()}") # Handle malformed lines
    except Exception as e:
        print(f"Error reading log file {log_path}: {e}")
    return {"logs": logs_out}


# --- API Endpoints ---
@app.get("/ping", summary="Health Check")
def ping() -> str:
  """Returns 'ok' if the server is running."""
  return "ok"

@app.get("/version", summary="Get Model Version")
def version() -> dict:
  """Returns the version of the regression model being used."""
  return {"version": model_version}

@app.get("/logs", summary="Get Request Logs")
def logs() -> dict:
  """Returns the content of the request log file."""
  return read_logs(log_path)

@app.post("/how_many_upvotes", summary="Predict Upvotes")
def how_many_upvotes(post: Post) -> dict:
    """
    Predicts the number of upvotes a Hacker News post might receive
    based on its title and timestamp.
    """
    start_time = datetime.now()

    # Get prediction using the loaded models and preprocessing
    prediction = predict_upvotes(post)

    end_time = datetime.now()
    latency_ms = (end_time - start_time).total_seconds() * 1000

    # Structure the log message
    log_message = {
        "latency_ms": round(latency_ms, 2),
        "version": model_version,
        "timestamp_utc": end_time.isoformat(), # Use ISO format for timestamp
        "input": post.dict(), # Use pydantic's dict() method
        "prediction": prediction,
    }

    # Log the request and prediction
    log_request(log_path, log_message)

    # Return the prediction
    return {"upvotes": prediction}


# --- Optional: Run directly with Uvicorn for testing ---
if __name__ == "__main__":
    import uvicorn
    print(f"Log file location: {log_path}")
    # Ensure log directory exists if running directly
    log_dir_path.mkdir(parents=True, exist_ok=True)
    # Note: Host '0.0.0.0' makes it accessible on the network (useful in containers)
    uvicorn.run(app, host="0.0.0.0", port=8000)