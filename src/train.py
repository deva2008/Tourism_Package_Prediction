
"""
Trains candidate models (DecisionTree, RandomForest, AdaBoost, GradientBoosting, XGBoost),
tunes hyperparameters, logs experiments (MLflow + JSON), evaluates on test set,
and registers the best model to the Hugging Face Model Hub.

Usage:
  HF_TOKEN=... HF_USERNAME=... HF_DATASET_REPO=... HF_MODEL_REPO=... python src/train.py
"""
import os, json, time
from pathlib import Path
import pandas as pd
import numpy as np

from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder, hf_hub_download
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib
import mlflow

from utils import load_env, repo_ids

TARGET = "ProdTaken"

def load_split_from_hf(dataset_id: str, token: str):
    train_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset",
                                 filename="processed/train.csv", token=token)
    test_path = hf_hub_download(repo_id=dataset_id, repo_type="dataset",
                                filename="processed/test.csv", token=token)
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    for df in [train_df, test_df]:
        df.columns = [c.strip() for c in df.columns]
    return train_df, test_df

def build_preprocessor(X: pd.DataFrame):
    numeric = [c for c in [
        "Age","NumberOfPersonVisiting","PreferredPropertyStar","NumberOfTrips",
        "NumberOfChildrenVisiting","MonthlyIncome","PitchSatisfactionScore",
        "NumberOfFollowups","DurationOfPitch"
    ] if c in X.columns]
    binary = [c for c in ["Passport","OwnCar"] if c in X.columns]
    categorical = [c for c in [
        "TypeofContact","CityTier","Occupation","Gender","MaritalStatus",
        "Designation","ProductPitched"
    ] if c in X.columns]

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc",  StandardScaler())]), numeric),
            ("bin", Pipeline([("imp", SimpleImputer(strategy="most_frequent"))]), binary),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh",  OneHotEncoder(handle_unknown="ignore"))]), categorical)
        ],
        remainder="drop"
    )
    return pre

def candidates():
    return {
        "DecisionTree": (DecisionTreeClassifier(), {
            "clf__max_depth": [3, 5, 7, None],
            "clf__min_samples_split": [2, 5, 10]
        }),
        "RandomForest": (RandomForestClassifier(n_estimators=200, n_jobs=-1), {
            "clf__max_depth": [None, 8, 12, 16],
            "clf__min_samples_split": [2, 5, 10]
        }),
        "AdaBoost": (AdaBoostClassifier(n_estimators=200), {
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0]
        }),
        "GradientBoosting": (GradientBoostingClassifier(), {
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [2, 3, 4]
        }),
        "XGBoost": (XGBClassifier(
            n_estimators=300, tree_method="hist", eval_metric="logloss", use_label_encoder=False
        ), {
            "clf__max_depth": [2, 3, 4, 5],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__subsample": [0.7, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 0.9, 1.0]
        })
    }

def evaluate(y_true, proba, thresh):
    pred = (proba >= thresh).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "auprc": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0))
    }

def main():
    env = load_env(strict=True)
    ids = repo_ids(env)
    token = env["HF_TOKEN"]

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("tourism_wellness")

    # Load HF splits
    train_df, test_df = load_split_from_hf(ids["dataset_id"], token)

    # Split train->train/val for threshold tuning
    X = train_df.drop(columns=[TARGET])
    y = train_df[TARGET].astype(int)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    pre = build_preprocessor(X)
    results = []
    best = {"name": None, "score": -1, "pipeline": None, "threshold": 0.5, "val_f1": -1}

    for name, (est, param_grid) in candidates().items():
        pipe = Pipeline([("pre", pre), ("clf", est)])
        # RandomizedSearchCV over grid converted to distributions by sampling
        # We'll just sample parameter combinations uniformly.
        params_list = []
        from itertools import product
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        for combo in product(*values):
            params_list.append(dict(zip(keys, combo)))

        # sample up to 20 combos for speed
        import random
        random.seed(42)
        random.shuffle(params_list)
        sample = params_list[:min(20, len(params_list))]

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model_name", name)

            best_f1 = -1
            best_t = 0.5
            best_estimator = None

            for p in sample:
                clf = Pipeline([("pre", pre), ("clf", est.__class__(**{
                    k.replace("clf__", ""): v for k, v in p.items()
                }))])
                clf.fit(X_tr, y_tr)
                proba_val = clf.predict_proba(X_val)[:, 1]

                # threshold sweep
                thresholds = np.linspace(0.1, 0.9, 41)
                for t in thresholds:
                    pred = (proba_val >= t).astype(int)
                    f1 = f1_score(y_val, pred, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_t = float(t)
                        best_estimator = clf

            mlflow.log_metric("val_best_f1", float(best_f1))
            mlflow.log_metric("val_best_threshold", float(best_t))
            for k, v in p.items():
                mlflow.log_param(k, v)

            results.append({"model": name, "val_best_f1": float(best_f1), "threshold": float(best_t)})
            if best_f1 > best["score"]:
                best.update({"name": name, "score": float(best_f1), "pipeline": best_estimator, "threshold": float(best_t)})

    # Evaluate best on test
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype(int)

    proba_test = best["pipeline"].predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, proba_test, best["threshold"])

    # Save artifacts locally
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], out_dir/"best_model.joblib")
    with open(out_dir/"threshold.json", "w") as f:
        json.dump({"threshold": best["threshold"]}, f)

    # Save experiment summary
    report = {
        "best_model": best["name"],
        "best_val_f1": best["score"],
        "threshold": best["threshold"],
        "test_metrics": metrics,
        "candidates": results,
        "timestamp": int(time.time())
    }
    Path("reports").mkdir(parents=True, exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    # Register model to HF Model Hub
    HfFolder.save_token(token)
    api = HfApi(token=token)
    create_repo(repo_id=ids["model_id"], repo_type="model", private=False, exist_ok=True, token=token)

    # Create a minimal model card
    card = f"""# Wellness Tourism Purchase Model
Best model: **{best['name']}**  
Validation F1: **{best['score']:.4f}**  
Test metrics: {json.dumps(metrics)}  
Threshold: **{best['threshold']:.2f}**
"""
    Path("models/README.md").write_text(card)

    upload_folder(
        folder_path="models",
        repo_id=ids["model_id"],
        repo_type="model",
        token=token,
        commit_message="Register best model + threshold"
    )

    print("[OK] Model registered to HF Model Hub:", ids["model_id"])
    print("[OK] Test metrics:", metrics)

if __name__ == "__main__":
    main()
