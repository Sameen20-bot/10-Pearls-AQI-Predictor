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

sl.set_page_config(page_title="AQI FORECAST KARACHI", page_icon="☀️", layout="wide")

API_URL = "https://one0-pearls-aqi-predictor.onrender.com"

with open("app.css", encoding="utf-8") as csss:
    sl.markdown(f"<style>{csss.read()}</style>", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10,4))
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


#Horizon Menu
selected = option_menu(
    menu_title = None,
    menu_icon="cast",
    default_index = 0,
    options = ["Home", "Dashboard", "Model Insights", "Analyze"],
    icons=["house", "bar-chart", "cpu", "upload"],
    orientation="horizontal",
     styles = {
        "container": {
            "background-color": "#007DCC",
            "border-radius": "0px",
            "padding": "6px",
            "box-shadow": "6px 6px 5px black",
        },
        "nav-link": {
            "color": "#0B1E3D",
            "font-size": "20px",
            "font-weight": "600",
            "--hover-color": "rgba(255, 255, 255, 0.35)",
            "border-radius": "0px",
            "margin-start": "10px"
        },
        "nav-link-selected": {
            "background-color": "#0B1E3D",
            "color": "#FFFFFF",
        },
        "icon": {
            "font-size": "20px",
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
        sl.stop()

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
    <div id="head-current">{current_aqi}</div>
    <div class="holder-aqi">
    <div class="small-circle" style="background-color: {color};"></div>
    <div id="category-current">{current_aqi_status["category"]}</div>
    </div>
    <div id="aqi-message-current">{current_aqi_status["message"]}</div>
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
    <div id="head-current">{temperature}°C</div>
    <div id="category-current">Temperature</div>
    </div>
    ''', unsafe_allow_html=True)

    sl.markdown("---")

    ## Day 1, 2 and 3 AQI Prediction display
    col4, col5, col6 = sl.columns(3)

    col4.markdown(f'''
    <div class="current-box-day">
    <div class="day-heading">Day 1 AQI</div>
    <div id="day-head">{day1["aqi"]}</div>
    <div class="holder-aqi">
    <div class="small-circle" style="background-color: {color1};"></div>
    <div id="day-category">{day1_aqi_status["category"]}</div>
    </div>
    <div id="day-aqi-message">{day1_aqi_status["message"]}</div>
    </div>
    ''', unsafe_allow_html=True)

    col5.markdown(f'''
    <div class="current-box-day">
    <div class="day-heading">Day 2 AQI</div>
    <div id="day-head">{day2["aqi"]}</div>
    <div class="holder-aqi">
    <div class="small-circle" style="background-color: {color2};"></div>
    <div id="day-category">{day2_aqi_status["category"]}</div>
    </div>
    <div id="day-aqi-message">{day2_aqi_status["message"]}</div>
    </div>
    ''', unsafe_allow_html=True)

    col6.markdown(f'''
    <div class="current-box-day">
    <div class="day-heading">Day 3 AQI</div>
    <div id="day-head">{day3["aqi"]}</div>
    <div class="holder-aqi">
    <div class="small-circle" style="background-color: {color3};"></div>
    <div id="day-category">{day3_aqi_status["category"]}</div>
    </div>
    <div id="day-aqi-message">{day3_aqi_status["message"]}</div>
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

    try:
        sl.markdown("<div id='trend-aqi-predictor'>LAST 7 DAYS AQI TREND</div>", unsafe_allow_html=True)
        history_aqi = history(10)
    except Exception as e:
        sl.markdown("<div id='trend-aqi-predictor'>LAST DAYS AQI TREND</div>", unsafe_allow_html=True)
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

    filter = sl.selectbox("Select days for historical trend", options = [10,20,30])

    if filter == 10:            
        history_aqi = history(10)
            
        df = pd.DataFrame(history_aqi["data"])
            
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
            
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        df = df[df.index <= pd.Timestamp.now()]
            
        ax.plot(df["aqi"])
        sl.pyplot(fig)

    elif filter == 20:            
        history_aqi = history(20)
            
        df = pd.DataFrame(history_aqi["data"])
            
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
            
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        df = df[df.index <= pd.Timestamp.now()]
            
        ax.plot(df["aqi"])
        sl.pyplot(fig)

    elif filter == 30:            
        history_aqi = history(30)
            
        df = pd.DataFrame(history_aqi["data"])
            
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
            
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
            
        df = df[df.index <= pd.Timestamp.now()]
            
        ax.plot(df["aqi"])
        sl.pyplot(fig)

                
            
    
 

    

