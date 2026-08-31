import os
import json
import joblib
import numpy as np
import pandas as pd
import hopsworks
from pathlib import Path
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import time

DROP_DAY2 = ["boundary_layer_height", "f48_boundary_layer_height",
             "wind_pollution_dispersion", "delta_boundary_layer_height"]

DROP_DAY3 = ["boundary_layer_height", "f72_boundary_layer_height",
             "wind_pollution_dispersion", "delta_boundary_layer_height"]

VALIDATION_SIZE = 150

# Step1: Connecting to Hopsworks
def connect_hopsworks():
    CWD = Path.cwd()
    PROJECT_ROOT = CWD.parent if CWD.name == "Training_Pipeline" else CWD

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

# Step2: Reading data from feature group
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
    df.index = df.index.tz_localize(None)

    return df

# Step3: Preparing the data for day One
def prepare_data_day1(df):
    target = "day_1_future_aqi"

    df = df.dropna(subset = [target])

    X = df.drop(columns = [target])
    Y = df[target]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle = False)

    return X_train, X_test, Y_train, Y_test

# Step4: Preparing the data for day Two and Three
def prepare_data_day2_3(df, horizon):
    target = f"day_{horizon}_future_aqi"

    df = df.dropna(subset=[target])

    drop_cols = DROP_DAY2 if horizon == 2 else DROP_DAY3

    X = df.drop(columns=[target])
    X = X.drop(columns=drop_cols)
    Y = df[target]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=False
    )

    return X_train, X_test, Y_train, Y_test

# Step5: Training the Day 1 XGB Boost Model
def day1_train(X_train, Y_train):
    model = XGBRegressor(
        n_estimators=1800,
        learning_rate=0.01,
        max_depth=6,
        min_child_weight=10,
        gamma=0.4,
        colsample_bytree=1.0,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    ) 

    model.fit(X_train, Y_train)
    return model


# Step6: Training the Day 2 and 3 from 5 Blend Models
def day2_day3_train(X_train, Y_train):
    # For kitty boost
    X_fit, X_validation = X_train.iloc[:-VALIDATION_SIZE], X_train.iloc[-VALIDATION_SIZE:]
    Y_fit, Y_validation = Y_train.iloc[:-VALIDATION_SIZE], Y_train.iloc[-VALIDATION_SIZE:]

    #XGBoost All Model
    xgb = XGBRegressor(n_estimators = 520, 
                         learning_rate = 0.033,
                         max_depth = 3, 
                         random_state = 42, 
                         colsample_bytree = 0.7, 
                         subsample = 0.7,
                         min_child_weight = 5,
                         gamma=0.1,
                         reg_alpha=0.1,
                         reg_lambda=1.5,
                        ) 

    xgb.fit(X_train, Y_train)

    #Kitty Boost
    cbr = CatBoostRegressor(iterations = 3000,
                        learning_rate = 0.02,
                        depth = 4,
                        l2_leaf_reg = 10,
                        random_seed = 42,
                        early_stopping_rounds = 100,
                        verbose = False,)

    cbr.fit(X_fit, Y_fit, eval_set = [(X_validation, Y_validation)], verbose = False)

    #Random Forest
    rf = RandomForestRegressor(
    n_estimators = 500,
    max_depth = 8,
    min_samples_leaf = 10,      
    max_features = 0.5,        
    random_state = 42,
    n_jobs = -1,
    )

    rf.fit(X_train, Y_train)

    # Standard Scaler Transformation for Ridge and SVM
    sc = StandardScaler()
    X_train_clean = sc.fit_transform(X_train)

    # Ridge and SVM
    ridge = Ridge(alpha = 1.0)
    ridge.fit(X_train_clean, Y_train)

    bagsvm = BaggingRegressor(estimator = SVR(kernel='rbf', C=100, epsilon=1.0),
                        n_estimators = 500,
                        max_samples = 0.25,
                        bootstrap = True,
                        random_state = 42,
                        n_jobs = -1)

    bagsvm.fit(X_train_clean, Y_train)

    return {
    "xgb": xgb,
    "catboost": cbr,
    "rf": rf,
    "ridge": ridge,
    "svm": bagsvm,
    "scaler": sc,
    }

# Step7: Blend Prediction
def predict_blend(models, X_test):
    X_test_clean = models["scaler"].transform(X_test)

    blend = (models["xgb"].predict(X_test)
             + models["catboost"].predict(X_test)
             + models["rf"].predict(X_test)
             + models["ridge"].predict(X_test_clean)
             + models["svm"].predict(X_test_clean)) / 5

    return blend

# Step8: Now get all the metrics
def metrics_get(Y_actual, Y_pred):
    return {
        "r2":   float(r2_score(Y_actual, Y_pred)),
        "mae":  float(mean_absolute_error(Y_actual, Y_pred)),
        "mse":  float(mean_squared_error(Y_actual, Y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(Y_actual, Y_pred))),
    }

# Step9: Now check Baseline Score
def baseline_R2_Score(X_test, Y_test):
    return float(r2_score(Y_test, X_test["us_aqi"]))

# Step10: Best Model Save Otherwise old model use
def should_register_model(project, model_name, new_r2):
    model_register = project.get_model_registry()

    try:
        old = model_register.get_best_model(model_name,"r2", "max")
        old_r2 = old.training_metrics["r2"]
    except:
        # If model is not existing then create it
        print(f"First Model: {model_name}")
        return True

    print(f"{model_name}: Old Model: {old_r2} \n New Model: {new_r2}")
    return new_r2 >= old_r2

# Step11: Saving Day1 Model
def register_day1_model(project, model, metrics, feature_names):
    r2 = metrics["r2"]

    if not should_register_model(project, "aqi_day1_model", r2):
        print("Model is not good do not register it")
        return False

    # If it is passed model is good so, register it
    model_dir = Path("aqi_day1_xgb")
    model_dir.mkdir(exist_ok=True)
    model.save_model(str(model_dir / "xgb_aqi_day1.json"))

    # Saving features of passed model
    with open(model_dir / "features.json", "w") as f:
        json.dump(list(feature_names), f)

    model_register = project.get_model_registry()

    m = model_register.python.create_model(
    name="aqi_day1_model",
    metrics=metrics,
    description="Day-1 ahead AQI forecast for Karachi. "
                "Tuned XGBoost, retrained daily from aqi_daily_day1.",
    )

    m.save(str(model_dir))
    
    print("Model registered successfully")
    return True

# Step11: Saving Day2 and 3 Model
def register_day2_day3_model(project, models, metrics, feature_names, horizon):
    r2 = metrics["r2"]

    if not should_register_model(project, f"aqi_day{horizon}_model", r2):
        print(f"Day {horizon} model bad, do not register")
        return False

    model_dir = Path(f"aqi_day{horizon}_blend")
    model_dir.mkdir(exist_ok=True)

    models["xgb"].save_model(str(model_dir / f"xgb_aqi_day{horizon}.json"))
    models["catboost"].save_model(str(model_dir / f"catboost_aqi_day{horizon}.cbm"))
    joblib.dump(models["rf"],     model_dir / f"rf_aqi_day{horizon}.joblib")
    joblib.dump(models["ridge"],  model_dir / f"ridge_aqi_day{horizon}.joblib")
    joblib.dump(models["svm"],    model_dir / f"svm_aqi_day{horizon}.joblib")
    joblib.dump(models["scaler"], model_dir / f"scaler_aqi_day{horizon}.joblib")

    with open(model_dir / "features.json", "w") as f:
        json.dump(list(feature_names), f)

    model_register = project.get_model_registry()
    m = model_register.python.create_model(
        name=f"aqi_day{horizon}_model",
        metrics=metrics,
        description=f"Day-{horizon} ahead AQI forecast for Karachi. "
                    f"5-model blend (XGBoost, CatBoost, RandomForest, Ridge, BaggingSVR). "
                    f"Retrained daily from aqi_daily_day{horizon}.",
    )
    m.save(str(model_dir))

    print(f"Day {horizon} model registered")
    return True











