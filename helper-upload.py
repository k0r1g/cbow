from huggingface_hub import HfApi
import os

repo_id = "Kogero/hackernews-titles"
files_to_upload = [
    ("corpus.pkl", "data/corpus.pkl"),
    ("title_tokens.pkl", "data/title_tokens.pkl"),
    ("scores.pkl", "data/scores.pkl"),
    ("words_to_ids.pkl", "tokenizer/words_to_ids.pkl"),
    ("ids_to_words.pkl", "tokenizer/ids_to_words.pkl"),
    ("title_token_ids.pkl", "data/title_token_ids.pkl"),
]

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

for local_file, repo_path in files_to_upload:
    print(f"Uploading {local_file} → {repo_path}")
    api.upload_file(
        path_or_fileobj=local_file,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="dataset",
        token=os.getenv("HF_TOKEN")
    )
