import os
import json
import joblib
import pandas as pd
import hopsworks
from pathlib import Path
from dotenv import load_dotenv
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

#Step1: Hopsworks connection function
def connect_hopsworks():
    CWD = Path.cwd()
    PROJECT_ROOT = CWD.parent if CWD.name == "App" else CWD
    
    load_dotenv(PROJECT_ROOT / ".env")
    
    key = os.getenv("HOPSWORK_KEY")
    
    project = hopsworks.login(
    project='aqi_karachi_samu_2026',
    host="eu-west.cloud.hopsworks.ai",
    port=443,
    api_key_value=key,
    )
    
    fs = project.get_feature_store()
    print(f"Connected to {project.name}")
    
    return project


#Step2: Day 1 model from Registry
def day1_model_load(project):
    model_register = project.get_model_registry()    

    model_meta = model_register.get_model("aqi_day1_model", version=1)
    path = model_meta.download()

    with open(f"{path}/features.json") as f:
        feature_names = json.load(f)

    model = XGBRegressor()
    model.load_model(f"{path}/xgb_aqi_day1.json")

    return model, feature_names


#Step2: Day 2 and 3 model from Registry
def load_blend_model(project, horizon):
    model_register = project.get_model_registry()
    model_meta = model_register.get_model(f"aqi_day{horizon}_model", version=1)
    path = model_meta.download()

    with open(f"{path}/features.json") as f:
        feature_names = json.load(f)

    xgb = XGBRegressor()
    xgb.load_model(f"{path}/xgb_aqi_day{horizon}.json")

    cbr = CatBoostRegressor()
    cbr.load_model(f"{path}/catboost_aqi_day{horizon}.cbm")

    rf     = joblib.load(f"{path}/rf_aqi_day{horizon}.joblib")
    ridge  = joblib.load(f"{path}/ridge_aqi_day{horizon}.joblib")
    bagsvm    = joblib.load(f"{path}/svm_aqi_day{horizon}.joblib")
    scaler = joblib.load(f"{path}/scaler_aqi_day{horizon}.joblib")

    models = {
        "xgb": xgb,
        "catboost": cbr,
        "rf": rf,
        "ridge": ridge,
        "svm": bagsvm,
        "scaler": scaler,
    }

    return models, feature_names
