# Tourism Package Prediction — Full MLOps on Hugging Face

This repo satisfies the **assignment criteria** end-to-end: data registration, preparation, model training with experiment tracking, model registration, Streamlit deployment, and a **GitHub Actions** workflow that automates it all.

## 🔐 Environment Variables
Create a `.env` (or export in CI Secrets) based on **env.example**:
- `HF_TOKEN`: Hugging Face access token (write)
- `HF_USERNAME`: your HF handle or org (e.g., `yourname`)
- `HF_DATASET_REPO`: dataset repo name (e.g., `tourism-wellness`)
- `HF_MODEL_REPO`: model repo name (e.g., `tourism-wellness-model`)
- `HF_SPACE_REPO`: space repo name (e.g., `tourism-wellness-app`)

## 📁 Structure
```
.
├─ data/                      # (local) raw dataset for initial registration
│  └─ tourism.csv
├─ src/
│  ├─ utils.py
│  ├─ data_register.py        # Create dataset repo + upload raw
│  ├─ prepare_data.py         # Load from HF, clean, split, upload processed
│  ├─ train.py                # Load processed from HF, tune, track, evaluate, register model
│  └─ deploy_space.py         # Create/Update Streamlit Space and upload app files
├─ app_streamlit/
│  └─ app.py                  # Streamlit UI (loads model from HF Hub)
├─ reports/                   # metrics.json, etc.
├─ tests/
│  └─ test_data.py
├─ .github/workflows/pipeline.yml
├─ requirements.txt           # full pipeline dependencies
├─ space_requirements.txt     # minimal inference deps for Space
├─ Dockerfile                 # optional: containerize Streamlit app locally
└─ README.md
```

## ✅ How each rubric item is addressed

### 1) Data Registration
- **Create master + data folder:** already present.
- **Register data on HF Dataset Space:** run `python src/data_register.py` (uses `data/tourism.csv`).

### 2) Data Preparation
- **Load directly from HF data space:** `src/prepare_data.py` downloads `raw/tourism.csv` from the dataset repo.
- **Clean & drop unnecessary columns:** removes `CustomerID`, trims strings, drops duplicates.
- **Train/Test split saved locally:** `data/train.csv`, `data/test.csv`.
- **Upload splits back to HF:** uploaded to `processed/train.csv` & `processed/test.csv` in dataset repo.

### 3) Model Building + Experiment Tracking
- **Load train/test from HF:** `src/train.py` pulls `processed` splits from dataset repo.
- **Define models & params:** DecisionTree, RandomForest, AdaBoost, GradientBoosting, **XGBoost** with sampled parameter grids.
- **Tune & log:** threshold sweep to maximize F1, **MLflow** logging + `reports/metrics.json`.
- **Evaluate:** ROC-AUC, AUPRC, Accuracy, Precision, Recall, F1 on test.
- **Register best model to HF Hub:** uploads `models/` folder (model + threshold + model card) to the Model Hub repo.

### 4) Model Deployment
- **Dockerfile:** runs Streamlit app (local/container).
- **Load saved model from HF Hub:** app pulls latest model via `snapshot_download`.
- **Inputs → DataFrame → Predict:** implemented in `app_streamlit/app.py`.
- **Dependencies:** `space_requirements.txt` for Spaces; `requirements.txt` for full pipeline.
- **Hosting script:** `src/deploy_space.py` creates/updates a **Streamlit** Space and pushes app files.

### 5) MLOps Pipeline with GitHub Actions
- **pipeline.yml in repo:** see `.github/workflows/pipeline.yml`.
- **Lists all steps:** Data Register → Prepare → Train/Tune/Eval/Register Model → Deploy Space.
- **Push all files to GitHub:** workflow commits `reports/`, `models/`, `mlruns/` back to `main`.
- **Automate end-to-end:** runs on push/PR and `workflow_dispatch`.
- **Auto-push updates to main:** included (uses default `GITHUB_TOKEN`).

### 6) Output Evaluation (for your submission)
- **GitHub:** push this repo; include link + screenshot of folder structure and the executed workflow run.
- **Streamlit on HF Spaces:** run `python src/deploy_space.py` after the model is registered; include the Space link + screenshot.

## 🛠️ Local quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Export env variables (or use .env + direnv)
export HF_TOKEN=...
export HF_USERNAME=...
export HF_DATASET_REPO=tourism-wellness
export HF_MODEL_REPO=tourism-wellness-model
export HF_SPACE_REPO=tourism-wellness-app

# 1) Register raw data to HF Dataset repo
python src/data_register.py

# 2) Prepare data (load from HF, clean, split, upload processed)
python src/prepare_data.py

# 3) Train, tune, evaluate, and register best model to HF Model Hub
python src/train.py

# 4) Deploy Streamlit Space
python src/deploy_space.py
```

## 📦 Container (optional)
```bash
docker build -t tourism-app .
docker run -it --rm -p 7860:7860 -e HF_USERNAME=$HF_USERNAME -e HF_MODEL_REPO=$HF_MODEL_REPO tourism-app
# open http://localhost:7860
```

## 🔗 Add links in your notebook
- HF Dataset: `https://huggingface.co/datasets/$HF_USERNAME/$HF_DATASET_REPO`
- HF Model: `https://huggingface.co/$HF_USERNAME/$HF_MODEL_REPO`
- HF Space (Streamlit): `https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE_REPO`
- GitHub repo: your repository URL
