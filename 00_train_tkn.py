#
#
#
import collections
import pickle
import psycopg2
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import pandas as pd
import os
import random
import numpy as np

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

#
# Connect to Postgres and fetch Hacker News data
#
conn = psycopg2.connect("postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki")
cur = conn.cursor()
cur.execute("""SELECT title, score FROM "hacker_news"."items" 
               WHERE title IS NOT NULL AND score IS NOT NULL;""")
data = cur.fetchall()
titles = [row[0] for row in data]
scores = [row[1] for row in data]
conn.close()

print(f"Loaded {len(titles)} titles from Hacker News")
print(f"Sample titles: {titles[:3]}")
print(f"Sample scores: {scores[:3]}")

#
# Preprocess titles into tokens
#
def preprocess(text: str) -> list[str]:
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
  words = [word for word in words if word not in stop_words]
  
  # Filter out very rare words by counting frequencies
  stats = collections.Counter(words)
  words = [word for word in words if stats[word] > 5]
  
  return words

#
# Process all titles
#
all_tokens = []
for title in titles:
  tokens = preprocess(title)
  all_tokens.extend(tokens)

corpus = all_tokens
print(f"Total tokens: {len(corpus)}")
print(f"Sample tokens: {corpus[:10]}")

# Save corpus for later use
with open('corpus.pkl', 'wb') as f: pickle.dump(corpus, f)

# Save title tokens and scores for regression model
title_tokens = [preprocess(title) for title in titles]
with open('title_tokens.pkl', 'wb') as f: pickle.dump(title_tokens, f)
with open('scores.pkl', 'wb') as f: pickle.dump(scores, f)

#
# Create lookup tables for word to ID mapping
#
def create_lookup_tables(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
  word_counts = collections.Counter(words)
  vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
  int_to_vocab = {ii+1: word for ii, word in enumerate(vocab)}
  int_to_vocab[0] = '<PAD>'
  vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
  return vocab_to_int, int_to_vocab

#
# Create token mappings
#
words_to_ids, ids_to_words = create_lookup_tables(corpus)
tokens = [words_to_ids[word] for word in corpus]
print(f"Vocabulary size: {len(words_to_ids)}")
print(f"Sample token IDs: {tokens[:10]}")

#
# Save token mappings
#
with open('words_to_ids.pkl', 'wb') as f: pickle.dump(words_to_ids, f)
with open('ids_to_words.pkl', 'wb') as f: pickle.dump(ids_to_words, f)

#
# Create tokenized titles with IDs for the regression model
#
title_token_ids = []
for tokens in title_tokens:
    # Convert each title's tokens to IDs, handle unknown words
    title_ids = [words_to_ids.get(word, 0) for word in tokens]  # Use 0 (<PAD>) for unknown words
    title_token_ids.append(title_ids)

with open('title_token_ids.pkl', 'wb') as f: pickle.dump(title_token_ids, f)

print("Tokenization complete. Files saved:")
print("- corpus.pkl: All tokens")
print("- title_tokens.pkl: Tokens per title")
print("- scores.pkl: Score values")
print("- words_to_ids.pkl: Word to ID mapping")
print("- ids_to_words.pkl: ID to word mapping")
print("- title_token_ids.pkl: Token IDs per title")

#
# Upload to Hugging Face
#
print("\nUploading to Hugging Face Hub...")

try:
    # Install huggingface_hub if not already installed
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Installing huggingface_hub...")
        # Use sys.executable to ensure pip installs for the correct python env
        import sys
        os.system(f"{sys.executable} -m pip install -q huggingface_hub")
        from huggingface_hub import HfApi
    
    # Initialize API - It will automatically use HF_TOKEN environment variable if set
    # No explicit login() call needed for server environments
    print("Initializing Hugging Face API (using HF_TOKEN if set)...")
    api = HfApi()
    
    # Create a new dataset repository
    repo_id = "Kogero/hackernews-titles"
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    
    # Upload each file
    files_to_upload = [
        ("corpus.pkl", "data/corpus.pkl"),
        ("title_tokens.pkl", "data/title_tokens.pkl"),
        ("scores.pkl", "data/scores.pkl"),
        ("words_to_ids.pkl", "tokenizer/words_to_ids.pkl"),
        ("ids_to_words.pkl", "tokenizer/ids_to_words.pkl"),
        ("title_token_ids.pkl", "data/title_token_ids.pkl")
    ]
    
    for local_file, repo_path in files_to_upload:
        print(f"Uploading {local_file} to {repo_path}...")
        api.upload_file(
            path_or_fileobj=local_file,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset"
        )
    
    print(f"\nSuccess! All files uploaded to: https://huggingface.co/datasets/{repo_id}")

except Exception as e:
    print(f"Error uploading to Hugging Face Hub: {e}")
    print("Please ensure HF_TOKEN environment variable is set correctly.")
    print("You can still use the local files for training.")
