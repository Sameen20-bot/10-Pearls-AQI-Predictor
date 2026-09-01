import io
import requests
import datetime as dt
import streamlit as sl
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

sl.set_page_config(page_title="AQI FORECAST KARACHI", page_icon="☀️", layout="wide")

API_URL = "https://one0-pearls-aqi-predictor.onrender.com"

with open("app.css") as csss:
    sl.markdown(f"<style>{csss.read()}</style>", unsafe_allow_html=True)


# End Points Requests

@sl.cache_data(ttl=300)
def forecast():
    response = requests.get(f"{API_URL}/predict", timeout=240)
    response.raise_for_status()
    return response.json()

@sl.cache_data(ttl=300)
def history(days):
    response = requests.get(f"{API_URL}/history?days={days}", timeout=240)
    response.raise_for_status()
    return response.json()

@sl.cache_data(ttl=300)
def metrics():
    response = requests.get(f"{API_URL}/metrics", timeout=240)
    response.raise_for_status()
    return response.json()

@sl.cache_data(ttl=300)
def history_predict(days):
    response = requests.get(f"{API_URL}/history?days={days}&include_predictions=true", timeout=240)
    response.raise_for_status()
    return response.json()

@sl.cache_data(ttl=300)
def temperature_latest():
    response = requests.get(f"{API_URL}/latest-temperature", timeout=240)
    response.raise_for_status()
    return response.json()


# Now first Page layout
sl.markdown("<h1 style='text-align: center;'>KARACHI AQI PREDICTOR</h1>", unsafe_allow_html=True)

## Data Fetching for Page One
try:
    with sl.spinner("Waking up the API..."):
        requests.get(f"{API_URL}/health", timeout=90)
    with sl.spinner("Waking up the API, please wait..."):
        data = forecast()
        temp = temperature_latest()
except Exception as e:
    sl.error(f"Could not load data {str(e)}")
    sl.stop()

current_aqi = data["current_aqi"]
current_aqi_status = data["current_aqi_status"]
temperature = temp["temperature"]

col1, col2, col3 = sl.columns(3)

## Status color extract
if current_aqi_status["colour"] == "yellow":
    color = "#E4DA72"
elif current_aqi_status["colour"] == "green":
    color = "#7EC151"
elif current_aqi_status["colour"] == "orange":
    color = "#FFA02E"
elif current_aqi_status["colour"] == "red":
    color = "#FF1700"
elif current_aqi_status["colour"] == "purple":
    color = "#9564DD"
elif current_aqi_status["colour"] == "maroon":
    color = "#8B2626"


col1.markdown(f'''
  <div>
  <h1>{current_aqi}</h1>
  <div class="small-circle" style="background-color: {color};">{current_aqi_status["colour"]}</div>
  <h3>{current_aqi_status["category"]}</h3>
  <p>{current_aqi_status["message"]}</P>
  </div>
''', unsafe_allow_html=True)


