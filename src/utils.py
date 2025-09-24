
import os, json
from pathlib import Path
from typing import Dict, Any

REQUIRED_ENV = [
    "HF_TOKEN",          # Your Hugging Face access token
    "HF_USERNAME",       # Your HF username or org
    "HF_DATASET_REPO",   # Dataset repo name, e.g., "tourism-wellness"
    "HF_MODEL_REPO",     # Model repo name, e.g., "tourism-wellness-model"
    "HF_SPACE_REPO"      # Space repo name, e.g., "tourism-wellness-app"
]

def load_env(strict: bool = True) -> Dict[str, str]:
    env = {k: os.environ.get(k, "") for k in REQUIRED_ENV}
    if strict:
        missing = [k for k, v in env.items() if not v]
        if missing:
            raise RuntimeError(f"Missing required env vars: {missing}. "
                               f"See env.example and export them before running.")
    return env

def repo_ids(env: Dict[str, str]) -> Dict[str, str]:
    user = env["HF_USERNAME"]
    return {
        "dataset_id": f"{user}/{env['HF_DATASET_REPO']}",
        "model_id":   f"{user}/{env['HF_MODEL_REPO']}",
        "space_id":   f"{user}/{env['HF_SPACE_REPO']}"
    }

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def dump_json(obj: Any, path: Path):
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
