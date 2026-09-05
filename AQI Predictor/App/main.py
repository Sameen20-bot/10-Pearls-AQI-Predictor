from datetime import timedelta

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
import pandas as pd, io
from api_functions import (connect_hopsworks, day1_model_load,
                               load_blend_model, get_data, get_latest_row,
                               predict_blend, aqi_category, today_karachi)

app = FastAPI(title="Karachi AQI Forecast API")

# Loading the models and the features

project = connect_hopsworks()

day1_model , day1_features = day1_model_load(project)
day2_model, day2_features = load_blend_model(project, 2)
day3_model, day3_features = load_blend_model(project, 3)

# Home End Point

@app.api_route("/", methods=["GET", "HEAD"])
def home():
    return{
        "message": "Karachi AQI Forecast API",
        "status": "running",
        "endpoints": ["/predict", "/latest-temperature",
                      "/history", "/predict-file", "/health", "/metrics"]
    }

# Health End Point

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {
        "status": "running",
        "day1_features": len(day1_features),
        "day2_features": len(day2_features),
        "day3_features": len(day3_features),
        "day1_model": "Tuned XGBoost",
        "day2_model": "5-model blend",
        "day3_model": "5-model blend",
    }

# Predict End Point

@app.api_route("/predict", methods=["GET", "HEAD"])
def predict():
    try:
        # Day 1
        day1 = get_data(project, "aqi_daily_day1")
        row1 = get_latest_row(day1)
        X1 = row1[day1_features]
        p1 = float(day1_model.predict(X1)[0])

        # Day 2
        day2 = get_data(project, "aqi_daily_day2")
        row2 = get_latest_row(day2)
        X2 = row2[day2_features]
        p2 = float(predict_blend(day2_model, X2)[0])

        # Day 3
        day3 = get_data(project, "aqi_daily_day3")
        row3 = get_latest_row(day3)
        X3 = row3[day3_features]
        p3 = float(predict_blend(day3_model, X3)[0])

        # Today AQI
        current = float(row1["us_aqi"].iloc[0])


        t1 = row1.index[0]
        t2 = row2.index[0]
        t3 = row3.index[0]

        age = int((today_karachi() - t1).days)

        return {
            "current_aqi_date": str(t1.date()),
            "data_age_days": age,
            "is_stale": age > 1,
            "current_aqi": round(current),
            "current_aqi_status": aqi_category(current),
            "forecast": [
                {
                    "day": 1,
                    "date": str((t1 + timedelta(days=1)).date()),
                    "aqi": round(p1),
                    "aqi_status": aqi_category(p1),
                },
                {
                    "day": 2,
                    "date": str((t2 + timedelta(days=2)).date()),
                    "aqi": round(p2),
                    "aqi_status": aqi_category(p2),
                },
                {
                    "day": 3,
                    "date": str((t3 + timedelta(days=3)).date()),
                    "aqi": round(p3),
                    "aqi_status": aqi_category(p3),
                },
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"Prediction failed {str(e)}"
        )

# Latest Temperature End Point

@app.api_route("/latest-temperature", methods=["GET", "HEAD"])
def current_temperature():
    try:
        day1 = get_data(project, "aqi_daily_day1")
        row1 = get_latest_row(day1)
        X1 = row1[day1_features]
        temp = float(X1["temperature_2m"].iloc[0])

        # For time
        time_AQI = row1.index[0]

        return{
            "temperature": round(temp),
            "date": str(time_AQI.date())
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"Temperature fetch failed {str(e)}"
        )

# History End Point

@app.api_route("/history", methods=["GET", "HEAD"])
def history(days: int = 30, include_predictions: bool = False):
    if days < 1 or days > 365:
        raise HTTPException(status_code=400, detail="days must be between 1 and 365")
    try:
        df = get_data(project, "aqi_daily_day1")
        df = df[df.index <= today_karachi()]
        recent = df.tail(days)

        data = [
            {"date": str(d.date()), "aqi": round(float(v), 1)}
            for d, v in zip(recent.index, recent["us_aqi"])
        ]

        if include_predictions:
            preds = day1_model.predict(recent[day1_features])
            actual_next = recent["day_1_future_aqi"].values

            for i, row in enumerate(data):
                row["predicted_next_day_aqi"] = round(float(preds[i]), 1)

                a = actual_next[i]
                row["actual_next_day_aqi"] = None if pd.isna(a) else round(float(a), 1)

        return {"days": days, "data": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History failed: {e}")

# Predict File End Point

@app.api_route("/predict-file", methods=["POST", "HEAD"])
async def predict_file(file: UploadFile=File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code = 400,
            detail = "Please upload a csv file only."
        )

    contents = await file.read()
    df_file = pd.read_csv(io.BytesIO(contents))

    required_columns = "time"

    if required_columns not in df_file.columns:
        raise HTTPException(
            status_code = 400,
            detail = f"Please upload time column"
        )
    
    if len(df_file) == 0:
        raise HTTPException(
            status_code = 400,
            detail = "No records"
        )
    
    if len(df_file) > 500:
        raise HTTPException(
            status_code = 400,
            detail = "Maximum 500 records allowed"
        )

    # Now date time cleaning
    df_file["time"] = pd.to_datetime(df_file["time"])
    
    if df_file["time"].dt.tz is not None:        
        df_file["time"] = df_file["time"].dt.tz_localize(None)

    df_file["time"] = df_file["time"].dt.normalize()    
    df_file = df_file.set_index("time")

    today = today_karachi()

    df_file = df_file[df_file.index < today]

    if len(df_file) == 0:
        raise HTTPException(
            status_code = 400,
            detail = "All the dates are today or in the future. Upload past dates only."
        )

    # Match with the feature store
    df = get_data(project, "aqi_daily_day1")

    try:
        df = df[(df.index < today) & (df["day_1_future_aqi"].notna())]
        
        match = df[df.index.isin(df_file.index)]
        
        if len(match) == 0:
            raise HTTPException(
                 status_code = 404,
                 detail = "Dates not found in the Feature Store"
            )
        predictions = day1_model.predict(match[day1_features])

        out = pd.DataFrame({
            "time": match.index.date,
            "actual_aqi": match["us_aqi"].round().values,
            "predicted_next_day_aqi": predictions.round(),
            "actual_next_day_aqi": match["day_1_future_aqi"].round().values,
        })
        output = out.to_csv(index=False)

        return StreamingResponse(
            io.StringIO(output),
            media_type='text/csv',
            headers={
                "Content-Disposition": "attachment; filename=prediction.csv"
            }
        )


    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"File prediction failed {str(e)}"
        )

# Get Metrics

@app.api_route("/metrics", methods=["GET", "HEAD"])
def get_metrics():
    try:
        model_registry = project.get_model_registry()
        metrics_output = {}

        for horizon in [1, 2, 3]:
            meta = model_registry.get_model(f"aqi_day{horizon}_model", version=1)
            meta_dict = dict( meta.training_metrics)
            meta_dict["baseline_gain"] = round(meta_dict["r2"]-meta_dict["baseline_r2"], 3)

            metrics_output[f"day_{horizon}"] = meta_dict

        return{
            "metrics_output": metrics_output
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code = 500,
            detail = f"Something went wrong {str(e)}"
        )
