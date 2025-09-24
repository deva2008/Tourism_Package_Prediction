
"""
Creates (or reuses) a Hugging Face Dataset repo and uploads the raw dataset.
Requires env vars (see env.example).

Usage:
  HF_TOKEN=... HF_USERNAME=... HF_DATASET_REPO=... python src/data_register.py
"""
from pathlib import Path
from huggingface_hub import HfApi, upload_file, create_repo, HfFolder
import os

from utils import load_env, repo_ids

def main():
    env = load_env(strict=True)
    ids = repo_ids(env)
    token = env["HF_TOKEN"]

    HfFolder.save_token(token)
    api = HfApi(token=token)

    # Create dataset repo if it doesn't exist
    try:
        create_repo(repo_id=ids["dataset_id"], repo_type="dataset", exist_ok=True, token=token)
        print(f"[OK] Dataset repo ensured: {ids['dataset_id']}")
    except Exception as e:
        print("[WARN] create_repo:", e)

    # Upload raw CSV into datasets repo under 'raw/'
    local_csv = Path("data/tourism.csv")
    assert local_csv.exists(), "Local data/tourism.csv not found"

    remote_path = "raw/tourism.csv"
    upload_file(
        path_or_fileobj=str(local_csv),
        path_in_repo=remote_path,
        repo_id=ids["dataset_id"],
        repo_type="dataset",
        token=token
    )
    print(f"[OK] Uploaded {local_csv} -> {ids['dataset_id']}/{remote_path}")

if __name__ == "__main__":
    main()
