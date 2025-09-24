
"""
Creates a Hugging Face Space (Streamlit) and uploads the Streamlit app files.
It expects the *model* to already be published on HF Model Hub.

Usage:
  HF_TOKEN=... HF_USERNAME=... HF_SPACE_REPO=... python src/deploy_space.py
"""
from huggingface_hub import HfApi, HfFolder, create_repo, upload_file
from pathlib import Path
import os

from utils import load_env, repo_ids

def main():
    env = load_env(strict=True)
    ids = repo_ids(env)
    token = env["HF_TOKEN"]

    HfFolder.save_token(token)
    api = HfApi(token=token)

    # Create (or ensure) Space repo
    create_repo(repo_id=ids["space_id"], repo_type="space", space_sdk="streamlit",
                exist_ok=True, token=token)

    # Upload Streamlit app
    upload_file(
        path_or_fileobj="app_streamlit/app.py",
        path_in_repo="app.py",
        repo_id=ids["space_id"],
        repo_type="space",
        token=token
    )

    # requirements for Space
    upload_file(
        path_or_fileobj="space_requirements.txt",
        path_in_repo="requirements.txt",
        repo_id=ids["space_id"],
        repo_type="space",
        token=token
    )

    print("[OK] Streamlit Space deployed:", ids["space_id"])
    print("Remember to set Space Secrets if needed.")

if __name__ == "__main__":
    main()
