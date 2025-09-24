
"""
Loads raw data directly from the HF dataset space, cleans it, splits into train/test,
saves locally, and uploads processed splits back to HF dataset repo.

Usage:
  HF_TOKEN=... HF_USERNAME=... HF_DATASET_REPO=... python src/prepare_data.py
"""
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi, HfFolder, upload_file
from sklearn.model_selection import train_test_split

from utils import load_env, repo_ids

TARGET = "ProdTaken"

def load_raw_from_hf(dataset_id: str, token: str) -> pd.DataFrame:
    # Download raw CSV from dataset repo
    csv_path = hf_hub_download(
        repo_id=dataset_id,
        repo_type="dataset",
        filename="raw/tourism.csv",
        token=token
    )
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Basic cleaning: drop obvious non-features, duplicates; coerce numeric-like columns
    drop_cols = ["CustomerID"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=c)

    # Standardize categorical whitespace
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    df = df.drop_duplicates()

    # Ensure target is present
    assert TARGET in df.columns, f"Missing target column: {TARGET}"
    return df

def split_and_save(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df[TARGET] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values

    train_path = out_dir/"train.csv"
    test_path = out_dir/"test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path

def upload_processed(train_path: Path, test_path: Path, dataset_id: str, token: str):
    api = HfApi(token=token)
    for local, remote in [(train_path, "processed/train.csv"),
                          (test_path,  "processed/test.csv")]:
        upload_file(
            path_or_fileobj=str(local),
            path_in_repo=remote,
            repo_id=dataset_id,
            repo_type="dataset",
            token=token
        )
        print(f"[OK] Uploaded {local} -> {dataset_id}/{remote}")

def main():
    env = load_env(strict=True)
    ids = repo_ids(env)
    token = env["HF_TOKEN"]

    # Authenticate
    HfFolder.save_token(token)

    # Load raw from HF
    df_raw = load_raw_from_hf(ids["dataset_id"], token)
    df_clean = clean_df(df_raw)

    # Split & save locally
    train_path, test_path = split_and_save(df_clean)
    print(f"[OK] Saved splits: {train_path}, {test_path}")

    # Upload processed splits to HF
    upload_processed(train_path, test_path, ids["dataset_id"], token)

if __name__ == "__main__":
    main()
