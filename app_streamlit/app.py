
import streamlit as st
from pathlib import Path
Path.home().joinpath('.streamlit').mkdir(parents=True, exist_ok=True)
import pandas as pd
import json
import joblib
from huggingface_hub import snapshot_download
from pathlib import Path
import os

st.set_page_config(page_title="Wellness Package Purchase Prediction", layout="centered")

HF_USERNAME = os.environ.get("HF_USERNAME", "")
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")
MODEL_ID = f"{HF_USERNAME}/{HF_MODEL_REPO}" if HF_USERNAME and HF_MODEL_REPO else None

@st.cache_resource
def load_model_from_hub():
    assert MODEL_ID is not None, "Set HF_USERNAME and HF_MODEL_REPO in environment"
    local_dir = snapshot_download(repo_id=MODEL_ID, repo_type="model", local_dir="hf_model")
    model = joblib.load(Path(local_dir)/"best_model.joblib")
    with open(Path(local_dir)/"threshold.json") as f:
        threshold = json.load(f).get("threshold", 0.5)
    return model, threshold

st.title("🧘 Wellness Tourism — Purchase Predictor")
st.write("This app loads the **best model** from the Hugging Face Model Hub and predicts the purchase probability.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", min_value=0, max_value=120, value=35)
        CityTier = st.text_input("CityTier", value="1")
        Occupation = st.text_input("Occupation", value="Salaried")
        Gender = st.text_input("Gender", value="Male")
        NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=0, max_value=20, value=2)
        PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=1, max_value=5, value=3)
        MaritalStatus = st.text_input("MaritalStatus", value="Married")
        NumberOfTrips = st.number_input("NumberOfTrips", min_value=0, max_value=50, value=2)
    with col2:
        Passport = st.number_input("Passport (0/1)", min_value=0, max_value=1, value=1)
        OwnCar = st.number_input("OwnCar (0/1)", min_value=0, max_value=1, value=1)
        NumberOfChildrenVisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=0)
        Designation = st.text_input("Designation", value="Executive")
        MonthlyIncome = st.number_input("MonthlyIncome", min_value=0, value=50000, step=1000)
        PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=0, max_value=10, value=7)
        ProductPitched = st.text_input("ProductPitched", value="Basic")
        NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=20, value=2)
        DurationOfPitch = st.number_input("DurationOfPitch", min_value=0, max_value=300, value=30)
        TypeofContact = st.text_input("TypeofContact", value="Company Invited")

    submitted = st.form_submit_button("Predict")

if submitted:
    model, threshold = load_model_from_hub()
    data = {
        "Age": Age, "TypeofContact": TypeofContact, "CityTier": CityTier, "Occupation": Occupation,
        "Gender": Gender, "NumberOfPersonVisiting": NumberOfPersonVisiting, "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus, "NumberOfTrips": NumberOfTrips, "Passport": Passport, "OwnCar": OwnCar,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting, "Designation": Designation, "MonthlyIncome": MonthlyIncome,
        "PitchSatisfactionScore": PitchSatisfactionScore, "ProductPitched": ProductPitched, "NumberOfFollowups": NumberOfFollowups,
        "DurationOfPitch": DurationOfPitch
    }
    X = pd.DataFrame([data])
    proba = float(model.predict_proba(X)[0, 1])
    label = int(proba >= threshold)

    st.subheader("Prediction")
    st.json({"purchase_probability": proba, "will_purchase": label, "threshold_used": threshold})
