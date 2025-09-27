---
pretty_name: Wellness Tourism — Customer Propensity
license: other
language: en
tags: [tabular, classification, propensity, tourism]
task_categories: [tabular-classification]
size_categories: [1K<n<10K]
---

# Wellness Tourism — Customer Propensity Dataset

**Task:** Binary classification (`ProdTaken`: 0/1) to predict purchase of the Wellness Package.

## Files
- `raw/tourism.csv`
- `processed/train.csv`, `processed/test.csv`

## Quick start
```python
from huggingface_hub import hf_hub_download
import pandas as pd
p = hf_hub_download("deva8217/tourism-wellness", "processed/train.csv", repo_type="dataset")
df = pd.read_csv(p)
df.head()
