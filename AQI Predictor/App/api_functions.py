import os
import json
import joblib
import time
import pandas as pd
import hopsworks
from pathlib import Path
from dotenv import load_dotenv
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import datetime as dt

DATA_CACHE = {}


KARACHI_TZ = dt.timezone(dt.timedelta(hours=5))


def today_karachi():
    return pd.Timestamp(dt.datetime.now(KARACHI_TZ).date())


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

# Step3: Reading data from feature group
def read_feature_group(project, fg_name):
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=fg_name, version=1)

    for attempt in range(3):
        try:
            df = fg.read()
            break
        except Exception as e:
            print(f"Read attempted: {attempt + 1},  failed because {str(e)}")
            if attempt == 2:
                raise
            time.sleep(30)

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").set_index("time")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.index = df.index.normalize()

    return df


# Step4: Getting data from feature group
def get_data(project, fg_name, max_age_minutes=30):
    recieved_time = time.time()

    if fg_name in DATA_CACHE:
        saved_data, saved_time = DATA_CACHE[fg_name]

        again_request_time = (recieved_time - saved_time) / 60

        if again_request_time < max_age_minutes:
            return saved_data

    df =  read_feature_group(project, fg_name)

    DATA_CACHE[fg_name] = (df,recieved_time)

    return df


# Step5: Getting latest row
def get_latest_row(df, time_col="time"):
    if df.empty:
        raise ValueError("Dataset is empty")

    df = df.copy()

    if time_col in df:
        df[time_col] = pd.to_datetime(df[time_col])

        if df[time_col].dt.tz is not None:        
            df[time_col] = df[time_col].dt.tz_localize(None)

        df = df.set_index(time_col)

    df = df.sort_index()

    df = df[df.index <= today_karachi()]

    if df.empty:
        raise ValueError("No rows on or before today in the feature store")

    return df.tail(1)
    

# Step6: Blend Prediction
def predict_blend(models, X_test):
    X_test_clean = models["scaler"].transform(X_test)

    blend = (models["xgb"].predict(X_test)
             + models["catboost"].predict(X_test)
             + models["rf"].predict(X_test)
             + models["ridge"].predict(X_test_clean)
             + models["svm"].predict(X_test_clean)) / 5

    return blend


# Step7: AQI category
def aqi_category(value):
    if value <= 50:
        category, colour = "Good", "green"
        message = "Air quality is good."

    elif value <= 100:
        category, colour = "Moderate", "yellow"
        message = "Air quality is acceptable."

    elif value <= 150:
        category, colour = "Unhealthy for Sensitive Groups", "orange"
        message = "Sensitive groups may experience health effects."

    elif value <= 200:
        category, colour = "Unhealthy", "red"
        message = "Everyone may feel effects. Limit time outdoors."

    elif value <= 300:
        category, colour = "Very Unhealthy", "purple"
        message = "Health alert: everyone may experience more serious health effects."

    else:
        category, colour = "Hazardous", "maroon"
        message = "Health emergency: everyone is likely to be affected."

    return {
        "category": category,
        "colour": colour,
        "alert": value > 150,
        "message": message,
    }
