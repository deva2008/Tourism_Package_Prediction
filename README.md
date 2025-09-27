# Wellness Tourism — Customer Propensity Dataset

**Task:** Binary classification (`ProdTaken`) to predict purchase of the Wellness Package.

## Files
- `raw/tourism.csv`
- `processed/train.csv`, `processed/test.csv`

## Usage
```python
from huggingface_hub import hf_hub_download
import pandas as pd
p = hf_hub_download("deva8217/tourism-wellness", "processed/train.csv", repo_type="dataset")
pd.read_csv(p).head()


### 3) Upload the README to your **HF Dataset** (no Markdown fences, use your real IDs)
```bash
python - <<'PY'
import os
from huggingface_hub import upload_file
rid = f"{os.environ.get('HF_USERNAME','deva8217')}/{os.environ.get('HF_DATASET_REPO','tourism-wellness')}"
tok = os.environ.get("HF_TOKEN")  # ok to be None if you are logged in & have write access
upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=rid,
    repo_type="dataset",
    token=tok,
)
print("✅ README uploaded to", rid)
PY

