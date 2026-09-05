import os
import json
import requests
import pandas as pd
import numpy as np
import hopsworks
from pathlib import Path
from dotenv import load_dotenv
import time


LATITUDE = 24.8608
LONGITUDE = 67.0104
PAST_DAYS = 92
FORECAST_DAYS = 7

sensible_Range_Karachi = {'temperature_2m':(0, 55),     
    'wind_speed_10m':(0, None),    
    'relative_humidity_2m':(0, 100),    
    'surface_pressure':(870, 1085), 
    'pm10':(0, None),
    'pm2_5':(0, None),
    'carbon_monoxide':(0, None),
    'nitrogen_dioxide':(0, None),
    'sulphur_dioxide':(0, None),
    'ozone':(0, None),
    'us_aqi':(0, None),
    'boundary_layer_height': (0, None),
    'dew_point_2m': (-20, 40),
    'precipitation': (0, None),
    'cloud_cover': (0, 100),
    'wind_direction_10m': (0, 360),
    'wind_gusts_10m': (0, None),
    'shortwave_radiation': (0, None),
    'dust': (0, None),
    'aerosol_optical_depth': (0, None),}

# Step1: Weather Fetching 

def weather_fetch():
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
        "hourly": ["temperature_2m", "wind_speed_10m", "relative_humidity_2m", "surface_pressure",
            "boundary_layer_height", "dew_point_2m", "precipitation", "cloud_cover",
            "wind_direction_10m", "wind_gusts_10m", "shortwave_radiation",],
        "timezone": "auto",
    }

    response = requests.get(url, params)

    data = response.json()

    weatherData = pd.DataFrame(data["hourly"])

    weatherData.time = pd.to_datetime(weatherData.time)

    return weatherData

# Step2: AQI Fetching 

def aqi_fetch():
    url2 = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params2 = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone","dust"
                , "aerosol_optical_depth","us_aqi"],
        "timezone": "auto",
        "past_days": PAST_DAYS,
        "forecast_days": FORECAST_DAYS,
    }

    response2 = requests.get(url2, params2)

    data2 = response2.json()

    aqiData = pd.DataFrame(data2["hourly"])

    aqiData.time = pd.to_datetime(aqiData.time)

    return aqiData

# Step3: Merging and Cleaning

def merge_and_clean(weather, aqi):
    df = pd.merge(weather, aqi, on="time")

    for col, (low,high) in sensible_Range_Karachi.items():    
        df[col] = df[col].clip(lower = low, upper = high)

    df = df.sort_values("time").reset_index(drop=True)

    df = df.set_index("time")

    return df


# Step4: Building hourly features

def building_hourly_features(df):
    # Baseline features
    df["hour"] = df.index.hour
    df["day"] = df.index.day
    df["month"] = df.index.month
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter

    # Other Derived Metrics
    df["previous_hour_aqi"] = df["us_aqi"].shift(1)
    df["previous_day_aqi"] = df["us_aqi"].shift(24)
    df["aqi_change_rate"] = df["us_aqi"].diff()

    # Statistical features
    rolling_mean_aqi = df["us_aqi"].shift(1)

    df["rolled_mean_aqi_6hr"] = rolling_mean_aqi.rolling(window = 6).mean()
    df["rolled_mean_aqi_24hr"] = rolling_mean_aqi.rolling(window = 24).mean()
    df["rolled_mean_aqi_72hr"] = rolling_mean_aqi.rolling(window = 72).mean()
    df["rolled_std_aqi_6hr"] = rolling_mean_aqi.rolling(window = 6).std()
    df["rolled_std_aqi_24hr"] = rolling_mean_aqi.rolling(window = 24).std()
    df["rolled_std_aqi_72hr"] = rolling_mean_aqi.rolling(window = 72).std()

    df["ewm_aqi_24hr"] = rolling_mean_aqi.ewm(span = 24).mean()
    df["ewm_std_24hr"] = rolling_mean_aqi.ewm(span = 24).std()

    # Future target features
    df["Day_1_Future_AQI"] = df["us_aqi"].shift(-24)
    df["Day_2_Future_AQI"] = df["us_aqi"].shift(-48)
    df["Day_3_Future_AQI"] = df["us_aqi"].shift(-72)

    # Future weather features
    weather_future = ["temperature_2m", "wind_speed_10m", "relative_humidity_2m", "surface_pressure",
		"boundary_layer_height", "dew_point_2m", "precipitation", "cloud_cover",
		 "wind_gusts_10m", "shortwave_radiation"]
    shifts = [24, 48, 72]

    for s in shifts:
        for w in weather_future:
            df[f"f{s}_{w}"] = df[w].shift(-s)

    # Domain features
    df["wind_pollution_dispersion"] =  df["wind_speed_10m"] * df["boundary_layer_height"]
    df["pressure_change_3hour"] =  df["surface_pressure"].diff(3)
    df["humidity_level"] = df["temperature_2m"] - df["dew_point_2m"]

    return df

# Step5: Building Daily features for Day 1

def building_day1_features(df):
    drop1 = [
    "Day_2_Future_AQI", "Day_3_Future_AQI",
    "f48_temperature_2m", "f48_wind_speed_10m", "f48_relative_humidity_2m",
    "f48_surface_pressure", "f48_boundary_layer_height", "f48_dew_point_2m",
    "f48_precipitation", "f48_cloud_cover", "f48_wind_gusts_10m", "f48_shortwave_radiation",
    "f72_temperature_2m", "f72_wind_speed_10m", "f72_relative_humidity_2m",
    "f72_surface_pressure", "f72_boundary_layer_height", "f72_dew_point_2m",
    "f72_precipitation", "f72_cloud_cover", "f72_wind_gusts_10m", "f72_shortwave_radiation",
    ]
    AQI_Data_Day1 = df.drop(columns = drop1)

    nan_ignore = ["boundary_layer_height", "f24_boundary_layer_height", "wind_pollution_dispersion"]
    nan_remove = [x for x in AQI_Data_Day1.columns if x not in nan_ignore]

    AQI_Data_Day1 = AQI_Data_Day1.dropna(subset = nan_remove)

    daily1 = AQI_Data_Day1.drop(columns=["Day_1_Future_AQI"]).resample("D").mean(numeric_only=True)
    daily1 = daily1.drop(columns=["hour"])
    daily1["Day_1_Future_AQI"] = daily1["us_aqi"].shift(-1)

    nan_ignore_day = ["boundary_layer_height", "f24_boundary_layer_height", "wind_pollution_dispersion","Day_1_Future_AQI"]
    nan_remove_day = [x for x in daily1.columns if x not in nan_ignore_day]

    daily1 = daily1.dropna(subset = nan_remove_day)

    daily1.columns = [c.lower() for c in daily1.columns]

    return daily1

# Step6: Building Daily features for Day 2 and 3

def building_day2_day3_features(df):
    drop2 = [
    "Day_1_Future_AQI", "Day_3_Future_AQI",
    "f24_temperature_2m", "f24_wind_speed_10m", "f24_relative_humidity_2m",
    "f24_surface_pressure", "f24_boundary_layer_height", "f24_dew_point_2m",
    "f24_precipitation", "f24_cloud_cover", "f24_wind_gusts_10m", "f24_shortwave_radiation",
    "f72_temperature_2m", "f72_wind_speed_10m", "f72_relative_humidity_2m",
    "f72_surface_pressure", "f72_boundary_layer_height", "f72_dew_point_2m",
    "f72_precipitation", "f72_cloud_cover", "f72_wind_gusts_10m", "f72_shortwave_radiation",
    ]

    AQI_Data_Day2 = df.drop(columns = drop2)

    nan_ignore2 = ["boundary_layer_height", "f48_boundary_layer_height", "wind_pollution_dispersion"]
    nan_remove2 = [x for x in AQI_Data_Day2.columns if x not in nan_ignore2]
    AQI_Data_Day2 = AQI_Data_Day2.dropna(subset = nan_remove2)

    daily2 = AQI_Data_Day2.drop(columns=["Day_2_Future_AQI"]).resample("D").mean(numeric_only=True)
    daily2 = daily2.drop(columns=["hour"])
    daily2["Day_2_Future_AQI"] = daily2["us_aqi"].shift(-2)

    nan_ignore_day2 = ["boundary_layer_height", "f48_boundary_layer_height", "wind_pollution_dispersion","Day_2_Future_AQI"]
    nan_remove_day2 = [x for x in daily2.columns if x not in nan_ignore_day2]

    daily2 = daily2.dropna(subset = nan_remove_day2)

    # daily2.to_csv(PROCESSED_DIR / "Day_Two_AQI.csv")

    # Cyclic patterns addition (low R2 score in Day 2 and Day3)
    daily2["cyclic_day_sin"] = np.sin(2 * np.pi * daily2.index.dayofyear / 365)
    daily2["cyclic_day_cos"] = np.cos(2 * np.pi * daily2.index.dayofyear / 365)

    daily2["cyclic_week_sin"] = np.sin(2 * np.pi * daily2.index.dayofweek / 7)
    daily2["cyclic_week_cos"] = np.cos(2 * np.pi * daily2.index.dayofweek / 7)

    # Delta weather additions to increase R2 more
    for v in ["temperature_2m", "wind_speed_10m", "surface_pressure",
            "relative_humidity_2m", "boundary_layer_height"]:
        daily2[f"delta_{v}"] = daily2["f48_" + v] - daily2[v]

    daily2 = daily2.drop(columns=["previous_hour_aqi", "aqi_change_rate",
                                "rolled_mean_aqi_6hr", "rolled_std_aqi_6hr",
                                "day", "quarter"], errors="ignore")

    drop3 = [
    "Day_1_Future_AQI", "Day_2_Future_AQI",
    "f24_temperature_2m", "f24_wind_speed_10m", "f24_relative_humidity_2m",
    "f24_surface_pressure", "f24_boundary_layer_height", "f24_dew_point_2m",
    "f24_precipitation", "f24_cloud_cover", "f24_wind_gusts_10m", "f24_shortwave_radiation",
    "f48_temperature_2m", "f48_wind_speed_10m", "f48_relative_humidity_2m",
    "f48_surface_pressure", "f48_boundary_layer_height", "f48_dew_point_2m",
    "f48_precipitation", "f48_cloud_cover", "f48_wind_gusts_10m", "f48_shortwave_radiation",
    ]

    AQI_Data_Day3 = df.drop(columns = drop3)

    nan_ignore3 = ["boundary_layer_height", "f72_boundary_layer_height", "wind_pollution_dispersion"]
    nan_remove3 = [x for x in AQI_Data_Day3.columns if x not in nan_ignore3]
    AQI_Data_Day3 = AQI_Data_Day3.dropna(subset = nan_remove3)

    daily3 = AQI_Data_Day3.drop(columns=["Day_3_Future_AQI"]).resample("D").mean(numeric_only=True)
    daily3 = daily3.drop(columns=["hour"])
    daily3["Day_3_Future_AQI"] = daily3["us_aqi"].shift(-3)

    nan_ignore_day3 = ["boundary_layer_height", "f72_boundary_layer_height", "wind_pollution_dispersion","Day_3_Future_AQI"]
    nan_remove_day3 = [x for x in daily3.columns if x not in nan_ignore_day3]

    daily3 = daily3.dropna(subset = nan_remove_day3)

    # daily3.to_csv(PROCESSED_DIR / "Day_Three_AQI.csv")

    # Cyclic patterns addition (low R2 score in Day 2 and Day3)
    daily3["cyclic_day_sin"] = np.sin(2 * np.pi * daily3.index.dayofyear / 365)
    daily3["cyclic_day_cos"] = np.cos(2 * np.pi * daily3.index.dayofyear / 365)

    daily3["cyclic_week_sin"] = np.sin(2 * np.pi * daily3.index.dayofweek / 7)
    daily3["cyclic_week_cos"] = np.cos(2 * np.pi * daily3.index.dayofweek / 7)

    # Delta weather additions to increase R2 more
    for v in ["temperature_2m", "wind_speed_10m", "surface_pressure",
            "relative_humidity_2m", "boundary_layer_height"]:
        daily3[f"delta_{v}"] = daily3["f72_" + v] - daily3[v]

    daily3 = daily3.drop(columns=["previous_hour_aqi", "aqi_change_rate",
                                "rolled_mean_aqi_6hr", "rolled_std_aqi_6hr",
                                "day", "quarter"], errors="ignore")

    daily2.columns = [c.lower() for c in daily2.columns]
    daily3.columns = [c.lower() for c in daily3.columns]
    
    return daily2, daily3

# Step7: Connect to Hopsworks

def connect_hopsworks():
    CWD = Path.cwd()
    PROJECT_ROOT = CWD.parent if CWD.name == "Feature_Pipeline" else CWD

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

def wait_for_job(fg, feature_name, max_wait_minutes=8):

    BUSY = ("RUNNING", "SUBMITTED", "ACCEPTED", "NEW", "NEW_SAVING",
            "INITIALIZING", "STARTING_APP_MASTER", "AGGREGATING_LOGS")

    deadline = time.time() + max_wait_minutes * 60

    while time.time() < deadline:
        try:
            state = str(fg.materialization_job.get_state()).upper()
        except Exception as e:
            # No job history yet, or the state call failed. Not worth
            # blocking the pipeline over, so carry on.
            print(f"{feature_name}: could not read job state - {e}")
            return

        if state not in BUSY:
            return

        print(f"{feature_name}: previous job is {state}, waiting 30s")
        time.sleep(30)

    print(f"{feature_name}: job still busy after {max_wait_minutes} min, continuing anyway")


# Step8: Current data pushes to Hopsworks
def push_to_hopsworks(project, daily1, daily2, daily3):
 
    fs = project.get_feature_store()

    DATASET_INSERT = {
        "aqi_daily_day1": daily1,
        "aqi_daily_day2": daily2,
        "aqi_daily_day3": daily3,
    }

    failed = []

    for feature_name, data in DATASET_INSERT.items():
        fg = fs.get_feature_group(name=feature_name, version=1)
        to_insert = data.reset_index()
        expected_latest = pd.to_datetime(to_insert["time"]).max()

        wait_for_job(fg, feature_name)

        inserted = False
        for attempt in range(3):
            try:
                fg.insert(to_insert, wait=False)
                inserted = True
                break
            except Exception as e:
                print(f"{feature_name}: attempt {attempt + 1} failed - {e}")
                time.sleep(20)

        if not inserted:
            print(f"{feature_name}: INSERT FAILED after 3 attempts")
            failed.append(feature_name)
            continue

     
        wait_for_job(fg, feature_name)
        time.sleep(20)

   
        try:
            check = fg.read()
            actual_latest = pd.to_datetime(check["time"]).max()

            if actual_latest.tz is not None:
                actual_latest = actual_latest.tz_localize(None)

            if actual_latest.normalize() < expected_latest.normalize():
                print(f"{feature_name}: NOT LANDED - sent up to "
                      f"{expected_latest.date()}, store only has {actual_latest.date()}")
                failed.append(feature_name)
            else:
                print(f"{feature_name}: verified up to {actual_latest.date()} "
                      f"({len(to_insert)} rows sent)")

        except Exception as e:
            print(f"{feature_name}: could not verify - {e}")

    if failed:
        raise RuntimeError(f"These feature groups did not update: {failed}")

    print("All three feature groups updated and verified")
