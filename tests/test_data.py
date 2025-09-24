
from pathlib import Path
import pandas as pd

def test_raw_dataset_present():
    assert Path("data/tourism.csv").exists(), "data/tourism.csv missing"

def test_target_exists():
    df = pd.read_csv("data/tourism.csv")
    assert "ProdTaken" in df.columns, "Target 'ProdTaken' not in dataset"
