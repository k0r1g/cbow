# cbow-helper-upload.py
# Uploads the Word2Vec model checkpoint from 01_train_w2v.py to Hugging Face Hub.

from huggingface_hub import HfApi, login
import os
from pathlib import Path

# --- Configuration ---
# IMPORTANT: Verify this path points to the actual checkpoint you want to upload!
# This path is based on the last run's logs (timestamp + epoch 2).
# Using the exact filename including the non-breaking hyphen:
LOCAL_CHECKPOINT_PATH = Path("checkpoints/2025_04_18__12_41_13.hn‑w2v‑ft.epoch2.word2vec")

REPO_ID = "Kogero/hackernews-word2vec"  # Target MODEL repository ID
PATH_IN_REPO = "model.word2vec"         # Standard name for the model file in the repo
REPO_TYPE = "model"
# --- End Configuration ---

print(f"Attempting to upload: {LOCAL_CHECKPOINT_PATH}")
print(f"                to: HF Repo {REPO_ID}/{PATH_IN_REPO} (type: {REPO_TYPE})")

# Check if the local file exists
if not LOCAL_CHECKPOINT_PATH.is_file():
    print(f"\nERROR: Local checkpoint file not found at '{LOCAL_CHECKPOINT_PATH}'")
    print("Please verify the LOCAL_CHECKPOINT_PATH variable in this script.")
    exit(1)

try:
    print("\nLogging in to Hugging Face Hub (if needed)...")
    # Attempt login - uses cached token or prompts if necessary
    # For non-interactive environments, ensure HF_TOKEN env var is set or logged in previously.
    login()

    print("Initializing HfApi...")
    api = HfApi()

    print(f"Ensuring repository '{REPO_ID}' exists...")
    api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True)

    print(f"\nUploading {LOCAL_CHECKPOINT_PATH.name} to {REPO_ID}/{PATH_IN_REPO}...")
    api.upload_file(
        path_or_fileobj=str(LOCAL_CHECKPOINT_PATH),
        path_in_repo=PATH_IN_REPO,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message=f"Upload fine-tuned model checkpoint: {LOCAL_CHECKPOINT_PATH.name}"
        # token=os.getenv("HF_TOKEN") # Implicitly uses cached token or env var via login()
    )

    print(f"\n✅ Success! Model checkpoint uploaded to: https://huggingface.co/{REPO_ID}")

except Exception as e:
    print(f"\n❌ An error occurred during upload: {e}")
    print("Please check your Hugging Face token/login and the file path.") 