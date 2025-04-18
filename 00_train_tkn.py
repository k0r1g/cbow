# 00_train_tkn.py – build Hacker‑News tokeniser + lookup tables
# -------------------------------------------------------------
import collections
import pickle
import re
import os
import sys
from pathlib import Path

import psycopg2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd                 # (still useful downstream)
import numpy as np

# ---------------------------------------------------------------------------
# 0️⃣  NLTK data (download once)
# ---------------------------------------------------------------------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# 1️⃣  Fetch titles + scores from Postgres
# ---------------------------------------------------------------------------
print("Connecting to Postgres …")
conn = psycopg2.connect(
    "postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki",  # <- your DSN
    application_name="hn-tokeniser",
)
cur = conn.cursor()
cur.execute(
    """SELECT title, score
         FROM "hacker_news"."items"
        WHERE title IS NOT NULL
          AND score IS NOT NULL;"""
)
rows = cur.fetchall()
conn.close()

titles  = [r[0] for r in rows]
scores  = [r[1] for r in rows]

print(f"Loaded {len(titles):,} titles from Hacker News")
print("Sample titles:", titles[:3])
print("Sample scores:", scores[:3])

# ---------------------------------------------------------------------------
# 2️⃣  Pre‑processing helpers
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"[^\w\s-]")   # chars to strip

def preprocess(text: str) -> list[str]:
    """Lower‑case, strip punctuation, split, drop stop‑words, keep alphabetic."""
    text = text.lower()
    text = TOKEN_RE.sub(" ", text).replace("-", " ")
    words = [w for w in word_tokenize(text) if w.isalpha() and w not in STOP_WORDS]
    return words

# ---------------------------------------------------------------------------
# 3️⃣  Build global corpus & per‑title tokens
# ---------------------------------------------------------------------------
all_tokens   = []
title_tokens = []

for t in titles:
    toks = preprocess(t)
    title_tokens.append(toks)
    all_tokens.extend(toks)

print(f"Total tokens (after cleaning): {len(all_tokens):,}")
print("First 10 tokens:", all_tokens[:10])

# ---------------------------------------------------------------------------
# 4️⃣  Global frequency filter  → keep TOP_N most frequent words
# ---------------------------------------------------------------------------
TOP_N = 50_000                     # match HF checkpoint (50 000 + <PAD>)
PAD   = "<PAD>"

freq   = collections.Counter(all_tokens)
most_common_words = [w for w, _ in freq.most_common(TOP_N)]

# Map unknowns to <PAD> (index 0)
idx_to_word = [PAD] + most_common_words
word_to_idx = {w: i for i, w in enumerate(idx_to_word)}

print(f"Vocabulary size (incl. {PAD}): {len(idx_to_word):,}")  # → 50 001

# ---------------------------------------------------------------------------
# 5️⃣  Convert corpus & titles to IDs
# ---------------------------------------------------------------------------
corpus_ids = [word_to_idx.get(w, 0) for w in all_tokens]  # 0 = <PAD>/UNK

title_token_ids = [
    [word_to_idx.get(w, 0) for w in toks] for toks in title_tokens
]

# ---------------------------------------------------------------------------
# 6️⃣  Persist artefacts
# ---------------------------------------------------------------------------
out = Path(".")
pickle.dump(all_tokens,            open(out/"corpus.pkl",          "wb"))
pickle.dump(title_tokens,          open(out/"title_tokens.pkl",    "wb"))
pickle.dump(scores,                open(out/"scores.pkl",          "wb"))
pickle.dump(word_to_idx,           open(out/"words_to_ids.pkl",    "wb"))
pickle.dump(idx_to_word,           open(out/"ids_to_words.pkl",    "wb"))
pickle.dump(title_token_ids,       open(out/"title_token_ids.pkl", "wb"))

print("\nTokenisation complete. Files saved:")
for fn in [
    "corpus.pkl", "title_tokens.pkl", "scores.pkl",
    "words_to_ids.pkl", "ids_to_words.pkl", "title_token_ids.pkl"
]:
    print("•", fn)

# ---------------------------------------------------------------------------
# 7️⃣  (Optional) upload to Hugging Face Hub
# ---------------------------------------------------------------------------
print("\nUploading to Hugging Face Hub …")
try:
    from huggingface_hub import HfApi
except ImportError:
    os.system(f"{sys.executable} -m pip install -q huggingface_hub")
    from huggingface_hub import HfApi

api     = HfApi()
repo_id = "Kogero/hackernews-titles"
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

files_to_upload = {
    "corpus.pkl":          "data/corpus.pkl",
    "title_tokens.pkl":    "data/title_tokens.pkl",
    "scores.pkl":          "data/scores.pkl",
    "words_to_ids.pkl":    "tokenizer/words_to_ids.pkl",
    "ids_to_words.pkl":    "tokenizer/ids_to_words.pkl",
    "title_token_ids.pkl": "data/title_token_ids.pkl",
}

for local, remote in files_to_upload.items():
    print(f"  ↳ uploading {local} → {remote}")
    api.upload_file(local, remote, repo_id=repo_id, repo_type="dataset")

print(f"\n✅  Done – dataset at https://huggingface.co/datasets/{repo_id}")
