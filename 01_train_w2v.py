# 01_train_w2v.py – fine‑tune HuggingFace CBOW Word2Vec on Hacker News tokens
# ---------------------------------------------------------------------------
# Robust loader version:  ➜ gracefully handles **three** formats
#   1. Full gensim Word2Vec pickle                → `Word2Vec.load()`
#   2. KeyedVectors pickle                        → `KeyedVectors.load()` then wrap
#   3. PyTorch state‑dict / nn.Module (.pth)      → build `KeyedVectors` from the
#      weight matrix (+ idx‑to‑word list) and wrap for incremental training.
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
import random
import pickle
import datetime
from pathlib import Path
from typing import List

import numpy as np
import tqdm
import wandb

from gensim.models import Word2Vec, KeyedVectors
from huggingface_hub import hf_hub_download

import torch  # needed for .pth loader

# Local project modules
import model        # provides CBOW & Regressor classes
import evaluate     # neighbour sanity‑checks
import dataset      # ensure HNTitles dataset is import‑able

# ---------------------------------------------------------------------------
# Reproducibility & run‑stamp
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

TS = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
RUN_NAME = f"{TS}.hn‑w2v‑ft"

# ---------------------------------------------------------------------------
# Config – HF model location & training hyper‑params
# ---------------------------------------------------------------------------
HF_REPO  = "Kogero/cbow-embeddings"   # change here if you move the checkpoint
HF_FILE  = "cbow_embeddings.pth"      # whatever you uploaded
HF_CACHE = "hf_models"                # local cache dir

HN_TOKENS_PATH = "title_tokens.pkl"
EPOCHS   = 2
BATCH_SIZE = 10_000

# ---------------------------------------------------------------------------
# Helper – build a gensim Word2Vec wrapper given vectors + vocab
# ---------------------------------------------------------------------------

def build_w2v_from_vectors(words: List[str], vectors: np.ndarray, window: int = 2):
    """Construct a **trainable** gensim Word2Vec from raw numpy vectors.
    This lets us keep using gensim's incremental training API."""
    kv = KeyedVectors(vector_size=vectors.shape[1])
    kv.add_vectors(words, vectors)

    w2v = Word2Vec(vector_size=vectors.shape[1], window=window, min_count=1, sg=0)
    # Build vocab with dummy counts so frequencies are >0
    w2v.build_vocab_from_freq({w: 1 for w in words})
    w2v.wv.vectors[:] = kv.vectors  # copy weights
    return w2v

# ---------------------------------------------------------------------------
# 1️⃣  Load Hacker News tokenised titles
# ---------------------------------------------------------------------------
with open(HN_TOKENS_PATH, "rb") as f:
    hn_corpus: list[list[str]] = pickle.load(f)

print(f"Loaded Hacker News corpus  –  {len(hn_corpus):,} titles")
print("Sample:", hn_corpus[:3])

# ---------------------------------------------------------------------------
# 2️⃣  Download + load the pre‑trained CBOW model (robust multi‑format)
# ---------------------------------------------------------------------------
print("\nFetching pre‑trained CBOW checkpoint from HuggingFace …")
model_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, cache_dir=HF_CACHE)
print("Checkpoint at:", model_path)

w2v: Word2Vec | None = None

# ---- A. try full gensim Word2Vec pickle ----------------------------------
try:
    w2v = Word2Vec.load(model_path, mmap="r")
    print("Loaded as gensim Word2Vec ✔️")
except Exception as e_word2vec:
    print("Not a Word2Vec pickle –", e_word2vec)

# ---- B. try KeyedVectors pickle ------------------------------------------
if w2v is None:
    try:
        kv = KeyedVectors.load(model_path, mmap="r")  # type: ignore
        print("Loaded as gensim KeyedVectors ✔️  (wrapping for training)")
        w2v = build_w2v_from_vectors(kv.index_to_key, kv.vectors, window=2)
    except Exception as e_kv:
        print("Not KeyedVectors –", e_kv)

# ---- C. try PyTorch .pth --------------------------------------------------
if w2v is None:
    print("Attempting to interpret as PyTorch state‑dict …")
    state = torch.load(model_path, map_location="cpu")

    vectors_np: np.ndarray | None = None
    idx2word: List[str] | None = None

    # Case 1: state is a torch.nn.Module
    if isinstance(state, torch.nn.Module):
        try:
            vectors_np = next(state.parameters()).detach().cpu().numpy()
            idx2word = getattr(state, "idx_to_word", None) or getattr(state, "itos", None)
            print("State‑dict is nn.Module – extracted first parameter as vectors.")
        except StopIteration:
            pass

    # Case 2: raw state‑dict
    if vectors_np is None and isinstance(state, dict):
        # Heuristic: find 2‑D tensor with largest 1st dimension
        weight_key = None
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.ndim == 2:
                weight_key = k
                break
        if weight_key is not None:
            vectors_np = state[weight_key].cpu().numpy()
            print(f"Using tensor '{weight_key}' as embedding matrix ({vectors_np.shape}).")

        # Find vocabulary list
        candidate_vocab_keys = [k for k in state if k.lower() in {"idx_to_word", "itos", "idx2word", "i2w"}]
        if candidate_vocab_keys:
            idx2word = state[candidate_vocab_keys[0]]
            print(f"Using vocab from key '{candidate_vocab_keys[0]}' (len={len(idx2word)}).")

    if vectors_np is not None and idx2word is None:
        try:
            with open("ids_to_words.pkl", "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    # detect if keys are **ints → words** (desired) or words → ints
                    if all(isinstance(k, int) for k in loaded.keys()):
                        # int ➔ str mapping – order by key
                        idx2word = [loaded[i] for i in range(max(loaded.keys()) + 1)]
                    else:
                        # words ➔ ints mapping – invert & order by index
                        inv = {v: k for k, v in loaded.items() if isinstance(v, int)}
                        idx2word = [inv[i] for i in range(max(inv.keys()) + 1)]
                elif isinstance(loaded, list):
                    idx2word = loaded
            print(f"✔️  Loaded idx2word from ids_to_words.pkl (len={len(idx2word)})")
        except Exception as e:
            raise RuntimeError("Could not locate idx_to_word in .pth file, and fallback ids_to_words.pkl failed.") from e

    if vectors_np is not None and idx2word is not None:
        w2v = build_w2v_from_vectors(list(idx2word), vectors_np, window=2)
        print("Successfully rebuilt gensim model from PyTorch checkpoint ✔️")
    else:
        raise RuntimeError("Could not locate both embedding weights and vocabulary in the .pth file.")

# ---------------------------------------------------------------------------
# 3️⃣  Sanity‑print model stats
# ---------------------------------------------------------------------------
print("Model ready:")
print("  vector_size =", w2v.vector_size)
print("  vocab        =", len(w2v.wv))
print("  window       =", w2v.window)
print("  sg           =", w2v.sg, "(0 = CBOW)")

# ---------------------------------------------------------------------------
# 4️⃣  Extend vocabulary with Hacker News words & fine‑tune (unchanged)
# ---------------------------------------------------------------------------
print("\nUpdating vocabulary with Hacker News tokens …")
orig_vocab_size = len(w2v.wv)
w2v.build_vocab(hn_corpus, update=True)
print(f"New vocab size: {len(w2v.wv):,}  (added {len(w2v.wv) - orig_vocab_size:,} tokens)")

# ---------------------------------------------------------------------------
# 5️⃣  wandb + training loop – unchanged from previous version
# ---------------------------------------------------------------------------
Path("checkpoints").mkdir(parents=True, exist_ok=True)

wandb.init(
    project="cbow",
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

try:
    ds = dataset.HNTitles()
    print(f"Dataset check – {len(ds):,} tokenised titles available.")
except Exception as exc:
    print("⚠️  Could not load dataset (continuing):", exc)

TOTAL_SENTS = len(hn_corpus)
for epoch in range(EPOCHS):
    random.shuffle(hn_corpus)
    prgs = tqdm.tqdm(range(0, TOTAL_SENTS, BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    for start in prgs:
        batch = hn_corpus[start : start + BATCH_SIZE]
        w2v.train(batch, total_examples=len(batch), epochs=1, compute_loss=True)
        loss = w2v.get_latest_training_loss()
        prgs.set_postfix({"loss": f"{loss:.4f}"})
        wandb.log({"loss": loss})

    ckpt_path = f"checkpoints/{RUN_NAME}.epoch{epoch+1}.word2vec"
    w2v.save(ckpt_path)
    art = wandb.Artifact("word2vec-weights", type="model")
    art.add_file(ckpt_path)
    wandb.log_artifact(art)

    cbow_wrap = model.CBOW(ckpt_path, trainable=False)
    evaluate.topk(cbow_wrap)
    print(f"Epoch {epoch+1} complete – loss {loss:.4f}\n")

for probe in ["ai", "python", "apple", "openai"]:
    cbow_wrap = model.CBOW(ckpt_path, trainable=False)
    evaluate.topk(cbow_wrap, word=probe)

wandb.finish()
print("\nTraining finished – final model saved in checkpoints/")

# ---------------------------------------------------------------------------
# 6️⃣  Upload to HF – unchanged
# ---------------------------------------------------------------------------
print("\nUploading model to HuggingFace Hub…")
try:
    from huggingface_hub import login, HfApi
except ImportError:
    os.system("pip install -q huggingface_hub")
    from huggingface_hub import login, HfApi

login()
api = HfApi()
repo_id = "Kogero/hackernews-word2vec"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

with open("README.md", "w") as f:
    f.write(f"""---
language:
- en
license: mit
tags:
- word2vec
- hacker-news
- embeddings
---
# Hacker News Word2Vec Embeddings (200‑D CBOW fine‑tuned)
Fine‑tuned on Hacker News titles starting from `{HF_REPO}/{HF_FILE}` for {EPOCHS} epoch(s).
Vector size = {w2v.vector_size}, window = {w2v.window}.""")

api.upload_file(path_or_fileobj="README.md", path_in_repo="README.md", repo_id=repo_id, repo_type="model")
api.upload_file(path_or_fileobj=ckpt_path, path_in_repo="model.word2vec", repo_id=repo_id, repo_type="model")
print(f"\n✅ Success! Model uploaded to: https://huggingface.co/{repo_id}")
