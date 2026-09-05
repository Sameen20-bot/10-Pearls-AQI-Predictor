import io
import datetime as dt
import requests
import datetime as dt
import streamlit as sl
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from streamlit_lottie import st_lottie
import json
from pathlib import Path
import streamlit.components.v1 as com
from streamlit_option_menu import option_menu


APP_DIR = Path(__file__).parent

sl.set_page_config(page_title="AQI FORECAST KARACHI", page_icon="🍃", layout="wide", initial_sidebar_state="expanded")

API_URL = "https://one0-pearls-aqi-predictor.onrender.com"

with open(APP_DIR / "app.css", encoding="utf-8") as csss:
    sl.markdown(f"<style>{csss.read()}</style>", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(24,10))
plt.style.use("https://raw.githubusercontent.com/dhaitz/matplotlib-stylesheets/master/pitayasmoothie-dark.mplstyle")


# Fallback loader

def fallback_loader(filename):
    with open(APP_DIR/"fallback"/filename, encoding="utf-8") as f:
        return json.load(f)


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

@sl.cache_data(ttl=300, show_spinner=False)
def predict_file(file_bytes, file_name):
    files = {"file": (file_name, file_bytes, "text/csv")}
    response = requests.post(f"{API_URL}/predict-file", files=files, timeout=240)

    if response.status_code != 200:
        raise RuntimeError(f"[{response.status_code}] {response.text[:300]}")

    return response.content


#Menu
with sl.sidebar:
    sl.markdown("""
        <div class="sidebar-brand">
            <div>
                <div class="sidebar-brand-icon">🍃</div>
                <div class="sidebar-brand-title">KARACHI AQI</div>
                <div class="sidebar-brand-subtitle">AIR QUALITY FORECAST</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Home", "Dashboard", "Model Insights", "Analyze"],
        icons=["house", "bar-chart", "cpu", "upload"],
        default_index=0,
        orientation="vertical",
        styles={ 
            "container": {
                "padding": "4px 0px",
                "background-color": "#FFFFFF",
                "color": "#0B1E3D",
                "border-radius": "0px",
                "box-shadow": "6px 6px 5px rgba(0,0,0,0.15) !important", 
            },
            "icon::before": {
                "font-size": "18px",
                "color": "#FFFFFF",
            },
            "icon::after": {
                "font-size": "18px",
                "color": "#0B1E3D",
            },
            "nav-link": {
                "background-color": "transparent",
                "color": "#0B1E3D",
                "font-size": "16px",
                "font-weight": "600",
                "border-radius": "12px",
                "margin": "5px 8px",
                "white-space": "nowrap"
            },
            "nav-link-selected": {
                "background-color": "#0B1E3D",
                "color": "#FFFFFF",            
            },
        } 
    ) 


if selected == "Home":

    # Now first Page layout
    sl.markdown("<h1 id='karachi-aqi-predictor'>KARACHI AQI PREDICTOR</h1>", unsafe_allow_html=True)

    ## Data Fetching for Page One
    try:
        with sl.spinner("Waking up the API..."):
            requests.get(f"{API_URL}/health", timeout=90)
        with sl.spinner("Waking up the API, please wait..."):
            data = forecast()
            temp = temperature_latest()
    except Exception as e:
        sl.info(f"Could not load data, using fallback")
        data = fallback_loader('predict.json')
        temp = fallback_loader('latest-temperature.json')

    current_aqi = data["current_aqi"]
    current_aqi_status = data["current_aqi_status"]
    temperature = temp["temperature"]

    day1 = data["forecast"][0]
    day2 = data["forecast"][1]
    day3 = data["forecast"][2]

    day1_aqi_status = data["forecast"][0]["aqi_status"]
    day2_aqi_status = data["forecast"][1]["aqi_status"]
    day3_aqi_status = data["forecast"][2]["aqi_status"]

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


    ## Day One Color Extractor
    if day1_aqi_status["colour"] == "yellow":
        color1 = "#E4DA72"
    elif day1_aqi_status["colour"] == "green":
        color1 = "#7EC151"
    elif day1_aqi_status["colour"] == "orange":
        color1 = "#FFA02E"
    elif day1_aqi_status["colour"] == "red":
        color1 = "#FF1700"
    elif day1_aqi_status["colour"] == "purple":
        color1 = "#9564DD"
    elif day1_aqi_status["colour"] == "maroon":
        color1 = "#8B2626"


    ## Day Two Color Extractor
    if day2_aqi_status["colour"] == "yellow":
        color2 = "#E4DA72"
    elif day2_aqi_status["colour"] == "green":
        color2 = "#7EC151"
    elif day2_aqi_status["colour"] == "orange":
        color2 = "#FFA02E"
    elif day2_aqi_status["colour"] == "red":
        color2 = "#FF1700"
    elif day2_aqi_status["colour"] == "purple":
        color2 = "#9564DD"
    elif day2_aqi_status["colour"] == "maroon":
        color2 = "#8B2626"


    ## Day Three Color Extractor
    if day3_aqi_status["colour"] == "yellow":
        color3 = "#E4DA72"
    elif day3_aqi_status["colour"] == "green":
        color3 = "#7EC151"
    elif day3_aqi_status["colour"] == "orange":
        color3 = "#FFA02E"
    elif day3_aqi_status["colour"] == "red":
        color3 = "#FF1700"
    elif day3_aqi_status["colour"] == "purple":
        color3 = "#9564DD"
    elif day3_aqi_status["colour"] == "maroon":
        color3 = "#8B2626"

    col1.markdown(f'''
    <div class="current-box">
    <div class="head-current">{current_aqi}</div>
    <div class="holder-aqi">
    <div class="small-circle" style="background-color: {color};"></div>
    <div class="category-current">{current_aqi_status["category"]}</div>
    </div>
    <div class="aqi-message-current">{current_aqi_status["message"]}</div>
    </div>
    ''', unsafe_allow_html=True)

    @sl.cache_data(ttl=3600)
    def load_lottie_file(filename):
        with open(APP_DIR / "animation" / filename, encoding="utf-8") as f:
            return json.load(f)

    with col2:
        anim = load_lottie_file("weather cloud animation.json")
        com.html(f"""
        <style>
        html, body {{
            margin: 0;
            padding: 0;
            background: transparent !important;
            overflow: hidden;
        }}
        #lottie {{
            width: 300px;
            height: 300px;
            margin: 0 auto;
        }}
        </style>
        <div id="lottie"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js"></script>
        <script>
        lottie.loadAnimation({{
            container: document.getElementById('lottie'),
            renderer: 'svg',
            loop: true,
            autoplay: true,
            animationData: {json.dumps(anim)}
        }});
        </script>
        """, height=300)


    col3.markdown(f'''
    <div class="current-box">
    <div class="head-current">{temperature}°C</div>
    <div class="category-current">Temperature</div>
    </div>
    ''', unsafe_allow_html=True)

    sl.markdown("---")

    ## Day 1, 2 and 3 AQI Prediction display
    col4, col5, col6 = sl.columns(3)

    col4.markdown(f'''
    <div class="current-box-day">
    <div class="day-heading">Day 1 AQI</div>
    <div class="day-head">{day1["aqi"]}</div>
    <div class="holder-aqi">
    <div class="small-circle" style="background-color: {color1};"></div>
    <div class="day-category">{day1_aqi_status["category"]}</div>
    </div>
    <div class="day-aqi-message">{day1_aqi_status["message"]}</div>
    </div>
    ''', unsafe_allow_html=True)

    col5.markdown(f'''
    <div class="current-box-day">
    <div class="day-heading">Day 2 AQI</div>
    <div class="day-head">{day2["aqi"]}</div>
    <div class="holder-aqi">
    <div class="small-circle" style="background-color: {color2};"></div>
    <div class="day-category">{day2_aqi_status["category"]}</div>
    </div>
    <div class="day-aqi-message">{day2_aqi_status["message"]}</div>
    </div>
    ''', unsafe_allow_html=True)

    col6.markdown(f'''
    <div class="current-box-day">
    <div class="day-heading">Day 3 AQI</div>
    <div class="day-head">{day3["aqi"]}</div>
    <div class="holder-aqi">
    <div class="small-circle" style="background-color: {color3};"></div>
    <div class="day-category">{day3_aqi_status["category"]}</div>
    </div>
    <div class="day-aqi-message">{day3_aqi_status["message"]}</div>
    </div>
    ''', unsafe_allow_html=True)

    ### Alert
    def show_alert(aqi_value, alert, category):
        if alert == True:
            sl.warning(f"🚨 {aqi_value} is {category} avoid going outside and try to wear mask for your safety.")


    show_alert(current_aqi, current_aqi_status["alert"], current_aqi_status["category"])
    show_alert(day1["aqi"], day1_aqi_status["alert"], day1_aqi_status["category"])
    show_alert(day2["aqi"], day2_aqi_status["alert"], day2_aqi_status["category"])
    show_alert(day3["aqi"], day3_aqi_status["alert"], day3_aqi_status["category"])

    ## Trend Chart
    sl.markdown("---")
    sl.markdown("<div id='trend-aqi-predictor'>LAST DAYS AQI TREND</div>", unsafe_allow_html=True)

    try:
        history_aqi = history(10)
    except Exception as e:
        history_aqi = fallback_loader("history.json")

    df = pd.DataFrame(history_aqi["data"])

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df[df.index <= pd.Timestamp.now()]

    ax.plot(df["aqi"])
    sl.pyplot(fig)


# Dashboard layout
if selected == "Dashboard":

    col7, col8= sl.columns([1, 1])

    col9 = sl.columns([1])[0]

    col7.markdown('''
      <div class="category-content">
        <h2>AQI Category</h2>
        <hr>
        <div class="holder-aqi">
        <div class="small-circle" style="background-color: green;"></div>
        <div class="day-category">Good</div>
        </div>

        <div class="holder-aqi">
        <div class="small-circle" style="background-color: yellow;"></div>
        <div class="day-category">Moderate</div>
        </div>

        <div class="holder-aqi">
        <div class="small-circle" style="background-color: orange;"></div>
        <div class="day-category">Unhealthy for Sensitive Groups</div>
        </div>

        <div class="holder-aqi">
        <div class="small-circle" style="background-color: red;"></div>
        <div class="day-category">Unhealthy</div>
        </div>

        <div class="holder-aqi">
        <div class="small-circle" style="background-color: purple;"></div>
        <div class="day-category">Very Unhealthy</div>
        </div>

        <div class="holder-aqi">
        <div class="small-circle" style="background-color: maroon;"></div>
        <div class="day-category">Hazardous</div>
        </div>
      </div>
    ''', unsafe_allow_html=True)

    @sl.cache_data(ttl=3600)
    def load_lottie_file(filename):
        with open(APP_DIR / "animation" / filename, encoding="utf-8") as f:
            return json.load(f)
    
    with col8:
      anim = load_lottie_file("Wumpus series - wumpus curious.json")
      com.html(f"""
        <style>
            html, body {{
             margin: 0;
             padding: 0;
             background: transparent !important;
             overflow: hidden;
            }}
            #lottie {{
             width: 300px;
             height: 300px;
             margin: 0 auto;
            }}
        </style>
        <div id="lottie"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js"></script>
        <script>
        lottie.loadAnimation({{
            container: document.getElementById('lottie'),
            renderer: 'svg',
            loop: true,
            autoplay: true,
            animationData: {json.dumps(anim)}
        }});
        </script>
        """, height=300)
      
    with col9:
        with sl.expander("What to Know?"):
            sl.markdown('''
                <div class="info-content">
                <div class="holder-aqi-info">
                <div class="head">Summary Statistics:</div>
                <div class="parag">Summary statistics are single numbers
                or short values used to describe and communicate the main features of a large dataset.</div>
                </div>
            
                <div class="holder-aqi-info">
                <div class="head">Correlation Heatmap:</div>
                <div class="parag">A correlation heatmap is a graphical, color-coded 
                representation of a correlation matrix that displays the correlation coefficients between multiple variables.</div>
                </div>
            
                <div class="holder-aqi-info">
                <div class="head"">Multicollinearity:</div>
                <div class="parag">Multicollinearity happens when two or more independent 
                (predictor) variables in a regression model are closely correlated with each other.</div>
                </div>
            
                <div class="holder-aqi-info">
                <div class="head">Seasonal Decomposition:</div>
                <div class="parag">Seasonal decomposition is a statistical method that splits a time 
                series dataset into distinct, individual components to make underlying patterns easier to analyze and forecast.</div>
                </div>
            
                <div class="holder-aqi-info">
                <div class="head">Trend:</div>
                <div class="parag">A trend is a general direction in which something is changing, 
                developing, or moving over time.</div>
                </div>
            
                <div class="holder-aqi-info">
                <div class="head">Seasonal:</div>
                <div class="parag">Seasonal means happening, used, or existing only during a particular time or period of the year.</div>
                </div>

                <div class="holder-aqi-info">
                <div class="head">ACF:</div>
                <div class="parag">ACF stands for the Autocorrelation Function, a statistical tool used in time series
                analysis to measure how a variable relates to past versions of itself over different time steps.</div>
                </div>
                </div>
                ''', unsafe_allow_html=True)

    sl.markdown("---")

    filter = sl.selectbox("SELECT DAYS FOR HISTORICAL TREND", options = [10,20,30])

    if filter == 10:            
        history_aqi = history(10)
            
        df = pd.DataFrame(history_aqi["data"])
            
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
            
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
                        
        ax.plot(df["aqi"])
        sl.pyplot(fig)

    elif filter == 20:            
        history_aqi = history(20)
            
        df = pd.DataFrame(history_aqi["data"])
            
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
            
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
                        
        ax.plot(df["aqi"])
        sl.pyplot(fig)

    elif filter == 30:            
        history_aqi = history(30)
            
        df = pd.DataFrame(history_aqi["data"])
            
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
            
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
                        
        ax.plot(df["aqi"])
        sl.pyplot(fig)


    ## EDA PLOT
    sl.markdown("---")

    plot = {
    "AQI Trend": "simple_plot.png",
    "ACF": "acf.png",
    "AQI Category": "aqi_category.png",
    "Correlation Heatmap": "correlation.png",
    "Seasonal": "seasonal.png",
    "Trend": "trend.png",
    "Residuals": "resid.png",
    "Missing Values": "missing_values.png",
    "Mean by Day of Week": "aqi_mean_day_of_week.png",
    "Yearly Max AQI": "year_max_aqi.png",
    "Yearly Min AQI": "year_min_aqi.png",
    }


    eda = sl.selectbox("SELECT PLOT FOR EDA", options = list(plot.keys()))
    sl.image(str(APP_DIR/"images"/plot[eda]),  use_container_width=True)


# Model Insights Page

if selected == "Model Insights":
    sl.markdown("<div id='metric-evaluate'>METRICS EVALUATION</div>", unsafe_allow_html=True)
    col10, col11, col12, col13, col14, col15 = sl.columns(6)

    ## Scores Layout
    try:
        with sl.spinner("Waking up the API..."):
            requests.get(f"{API_URL}/health", timeout=90)
        with sl.spinner("Waking up the API, please wait..."):
            data = metrics()
    except Exception as e:
        sl.info(f"Could not load data, using fallback")
        data = fallback_loader('metrics.json')

    ### Day One Metrics
    day1_R2 = data["metrics_output"]["day_1"]["r2"]
    day1_Mae = data["metrics_output"]["day_1"]["mae"]
    day1_Rmse = data["metrics_output"]["day_1"]["rmse"]
    day1_Baseline_r2 = data["metrics_output"]["day_1"]["baseline_r2"]
    day1_Mse = data["metrics_output"]["day_1"]["mse"]
    day1_Baseline_Gain = data["metrics_output"]["day_1"]["baseline_gain"]

    ### Day Two Metrics
    day2_R2 = data["metrics_output"]["day_2"]["r2"]
    day2_Mae = data["metrics_output"]["day_2"]["mae"]
    day2_Rmse = data["metrics_output"]["day_2"]["rmse"]
    day2_Baseline_r2 = data["metrics_output"]["day_2"]["baseline_r2"]
    day2_Mse = data["metrics_output"]["day_2"]["mse"]
    day2_Baseline_Gain = data["metrics_output"]["day_2"]["baseline_gain"]

    ### Day Three Metrics
    day3_R2 = data["metrics_output"]["day_3"]["r2"]
    day3_Mae = data["metrics_output"]["day_3"]["mae"]
    day3_Rmse = data["metrics_output"]["day_3"]["rmse"]
    day3_Baseline_r2 = data["metrics_output"]["day_3"]["baseline_r2"]
    day3_Mse = data["metrics_output"]["day_3"]["mse"]
    day3_Baseline_Gain = data["metrics_output"]["day_3"]["baseline_gain"]

    col10.markdown(f'''
        <div class="metric-box-day">
        <div class="metric-head">Day 1 R2</div>
        <div class="number-sub">{round(day1_R2,2)}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    col11.markdown(f'''
        <div class="metric-box-day">
        <div class="metric-head">Day 1 MAE</div>
        <div class="number-sub">{round(day1_Mae,2)}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    col12.markdown(f'''
       <div class="metric-box-day">
       <div class="metric-head">Day 1 RMSE</div>
       <div class="number-sub">{round(day1_Rmse,2)}</div>
       </div>
        ''', unsafe_allow_html=True)
    
    col13.markdown(f'''
        <div class="metric-box-day">
        <div class="metric-head">Day 1 BASELINE R2</div>
        <div class="number-sub">{round(day1_Baseline_r2,2)}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    col14.markdown(f'''
       <div class="metric-box-day">
       <div class="metric-head">Day 1 MSE</div>
       <div class="number-sub">{round(day1_Mse,2)}</div>
       </div>
        ''', unsafe_allow_html=True)
    
    col15.markdown(f'''
       <div class="metric-box-day">
       <div class="metric-head">Day 1 BASELINE GAIN</div>
       <div class="number-sub">{round(day1_Baseline_Gain,2)}</div>
       </div>
        ''', unsafe_allow_html=True)

    
    col10.markdown(f'''
        <div class="metric-box-day">
        <div class="metric-head">Day 2 R2</div>
        <div class="number-sub">{round(day2_R2,2)}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    col11.markdown(f'''
        <div class="metric-box-day">
        <div class="metric-head">Day 2 MAE</div>
        <div class="number-sub">{round(day2_Mae,2)}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    col12.markdown(f'''
       <div class="metric-box-day">
       <div class="metric-head">Day 2 RMSE</div>
       <div class="number-sub">{round(day2_Rmse,2)}</div>
       </div>
        ''', unsafe_allow_html=True)
    
    col13.markdown(f'''
        <div class="metric-box-day">
        <div class="metric-head">Day 2 BASELINE R2</div>
        <div class="number-sub">{round(day2_Baseline_r2,2)}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    col14.markdown(f'''
       <div class="metric-box-day">
       <div class="metric-head">Day 2 MSE</div>
       <div class="number-sub">{round(day2_Mse,2)}</div>
       </div>
        ''', unsafe_allow_html=True)
    
    col15.markdown(f'''
       <div class="metric-box-day">
       <div class="metric-head">Day 3 BASELINE GAIN</div>
       <div class="number-sub">{round(day2_Baseline_Gain,2)}</div>
       </div>
        ''', unsafe_allow_html=True)

    
    col10.markdown(f'''
        <div class="metric-box-day">
        <div class="metric-head">Day 3 R2</div>
        <div class="number-sub">{round(day3_R2,2)}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    col11.markdown(f'''
        <div class="metric-box-day">
        <div class="metric-head">Day 3 MAE</div>
        <div class="number-sub">{round(day3_Mae,2)}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    col12.markdown(f'''
       <div class="metric-box-day">
       <div class="metric-head">Day 3 RMSE</div>
       <div class="number-sub">{round(day3_Rmse,2)}</div>
       </div>
        ''', unsafe_allow_html=True)
    
    col13.markdown(f'''
        <div class="metric-box-day">
        <div class="metric-head">Day 3 BASELINE R2</div>
        <div class="number-sub">{round(day3_Baseline_r2,2)}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    col14.markdown(f'''
       <div class="metric-box-day">
       <div class="metric-head">Day 3 MSE</div>
       <div class="number-sub">{round(day3_Mse,2)}</div>
       </div>
        ''', unsafe_allow_html=True)
    
    col15.markdown(f'''
       <div class="metric-box-day">
       <div class="metric-head">Day 3 BASELINE GAIN</div>
       <div class="number-sub">{round(day3_Baseline_Gain,2)}</div>
       </div>
        ''', unsafe_allow_html=True)


    ## Actual Vs Predicted Chart

    
    ## Model Insights Diagram
    sl.markdown("---")
    sl.image(str(APP_DIR/"images"/"model_analysis.png"),  use_container_width=True)

    ## What to Know
    sl.markdown("---")

    with sl.expander("What to Know?"):
         sl.markdown('''
             <div class="info-content">
             <div class="holder-aqi-info">
             <div class="head">Business Problem:</div>
             <div class="parag">The real-world need the project solves. Here: Karachi 
             residents need to know if the air will be unsafe in the next 3 days, so they can plan outdoor activity in advance.</div>
             </div>
                        
             <div class="holder-aqi-info">
             <div class="head">Data Collection:</div>
             <div class="parag">Gathering raw data from a source. Here: 3 years of hourly
             weather and pollutant readings for Karachi from the Open-Meteo API.</div>
             </div>
                        
             <div class="holder-aqi-info">
             <div class="head"">Data Cleaning:</div>
             <div class="parag">Fixing bad values before modelling — sensor errors, impossible readings, missing gaps.
              Here: domain-based bounds  (like humidity capped at 0-100%) instead of statistical methods, 
              so real pollution spikes were preserved.</div>
             </div>
                        
             <div class="holder-aqi-info">
             <div class="head">Feature Engineering:</div>
             <div class="parag">Turning raw columns into inputs a model can learn from. 
             Here: time features, lag features (yesterday's AQI), rolling averages, and cyclic day/week encodings.</div>
             </div>
                        
             <div class="holder-aqi-info">
             <div class="head">Exploratory Data Analysis:</div>
             <div class="parag">Studying the data with charts and statistics before modelling, to find patterns and decide what features matter. 
             Here: it revealed strong winter/summer seasonality and confirmed that past AQI predicts future AQI.</div>
             </div>
                        
             <div class="holder-aqi-info">
             <div class="head">Feature Store:</div>
             <div class="parag">A central place where prepared features are stored and versioned, so training and live prediction
              always use exactly the same data. Here: Hopsworks, with one feature group per forecast horizon.</div>
             </div>
            
             <div class="holder-aqi-info">
             <div class="head">Model Evaluation And Training:</div>
             <div class="parag">Teaching models on past data and measuring how well they predict unseen data. Here: a chronological 80/20 split 
             (never shuffled, since shuffling would leak the future into training), scored with R², MAE and RMSE against a persistence baseline.</div>
             </div>

             <div class="holder-aqi-info">
             <div class="head">Model Registry:</div>
             <div class="parag">A versioned store for trained models and everything needed to reuse them — the scaler, the feature order,
              and the metrics. Here: three models registered in Hopsworks, each verified by reloading it and reproducing the exact predictions.</div>
             </div>

             <div class="holder-aqi-info">
             <div class="head">Automation (CI/CD):</div>
             <div class="parag">Scheduled workflows that run the pipelines without anyone pressing a button.
              Here: GitHub Actions fetches new data hourly and retrains daily, so the system stays current on its own.</div>
             </div>

             <div class="holder-aqi-info">
             <div class="head">Serving and Monitoring:</div>
             <div class="parag">Making the model available to users and checking that it stays up. Here: a FastAPI service on Render 
             delivers predictions to the Streamlit dashboard, with UptimeRobot pinging the health endpoint every 5 minutes.</div>
             </div>
             </div>
             ''', unsafe_allow_html=True)


    ## SHAP Analysis
    sl.markdown("---")
    
    plots = {
    "Day One Waterfall": "waterfall-day1.png",
    "Day Two Waterfall": "waterfall-day2.png",
    "Day Three Waterfall": "waterfall-day3.png",
    "Day One Beeswarm": "beeswarm-day1.png",
    "Day Two Beeswarm": "beeswarm-day2.png",
    "Day Three Beeswarm": "beeswarm-day3.png",
    }
    
    shap = sl.selectbox("SHAP MODEL ANALYSIS", options = list(plots.keys()))
    sl.image(str(APP_DIR/"images"/plots[shap]),  use_container_width=True)
               

# Analyze Page
if selected == "Analyze":
    file = sl.file_uploader("PLEASE UPLOAD THE FILE OF KARACHI HISTORICAL DATA", type = [".csv"])

    with sl.expander("How To Use?"):
        sl.markdown('''
        <div class="info-content">
        <div class="holder-aqi-info">
        <div class="parag">1. Upload a CSV with a time column only (max 500 dates).</div>
        </div>
    
        <div class="holder-aqi-info">
        <div class="parag">2. Dates must be between 2023-08-01 and today.</div>
        </div>
    
        <div class="holder-aqi-info">
        <div class="parag">3. Model will predict what tomorrow's AQI was for each date.</div>
        </div>
        </div>
        ''', unsafe_allow_html=True)

    try:
        if file is None:
            sl.info("Upload a CSV to get started.")
        else:
            try:
                with sl.spinner("Waking up the API..."):
                    requests.get(f"{API_URL}/health", timeout=90)

                with sl.spinner("Running predictions..."):
                    csv_bytes =  predict_file(file.getvalue(), file.name)

                sl.success("Predictions ready")

                sl.dataframe(pd.read_csv(io.BytesIO(csv_bytes)), use_container_width=True)

                sl.download_button(
                    label="Download Predictions CSV",
                    data= csv_bytes,
                    file_name="prediction.csv",
                    mime="text/csv",
                )

            except Exception as e:
                sl.error(f"Failed to predict the file: {str(e)}")
    except Exception as e:
            sl.error(f"Failed to predict the file {str(e)}")
            




                
            
    
 

    

