"""
predict.py  –  Called by Node.js via child_process.spawn
Usage:  python predict.py '<json_input>'
Output: JSON  { "cases": 27.4, "risk": "HIGH", "message": "..." }
"""

import sys
import json
import numpy as np
import joblib
from datetime import datetime

import os as _os
MODEL_PATH     = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "models", "hospital_model.pkl")
CITY_DATA_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "models", "city_data.json")

FEATURES = [
    "AQI", "PM2.5", "PM10", "NO2", "SO2",
    "Elevation", "Slope", "TPI", "Industry",
    "AQI_lag1", "AQI_lag3", "AQI_lag5", "AQI_lag7",
    "PM2.5_lag3", "PM2.5_lag7",
    "AQI_roll3", "AQI_roll5", "AQI_roll7", "PM2.5_roll7",
    "Month", "Day_of_week", "Winter", "Summer", "Monsoon"
]

def classify(cases):
    if cases < 10:
        return "LOW",    "Normal hospital operations expected."
    elif cases <= 25:
        return "MEDIUM", "Prepare moderate respiratory care capacity."
    else:
        return "HIGH",   "Possible surge in respiratory admissions. Hospitals should prepare additional resources."

def compute_rolling(aqi, lag1, lag3, lag5, lag7):
    """Approximate rolling averages from available lag values."""
    roll3 = np.mean([aqi, lag1, lag3])
    roll5 = np.mean([aqi, lag1, lag3, lag5, (lag3+lag5)/2])
    roll7 = np.mean([aqi, lag1, lag3, lag5, lag7, (lag5+lag7)/2, (lag3+lag5)/2])
    return roll3, roll5, roll7

def compute_pm25_roll7(pm25, pm25_lag3, pm25_lag7):
    return np.mean([pm25, pm25_lag3, pm25_lag7,
                    (pm25+pm25_lag3)/2, (pm25_lag3+pm25_lag7)/2,
                    pm25_lag3, pm25_lag7])

def get_seasonal(month):
    """
    IMD India Season Definitions (India Meteorological Department)
    Winter:       December(12), January(1), February(2)  -> winter=1
    Pre-Monsoon:  March(3), April(4), May(5)              -> summer=1 (model feature)
    SW Monsoon:   June(6), July(7), August(8), Sep(9)    -> monsoon=1
    Post-Monsoon: October(10), November(11)              -> all zeros
    """
    winter  = 1 if month in [12, 1, 2]    else 0
    summer  = 1 if month in [3, 4, 5]     else 0
    monsoon = 1 if month in [6, 7, 8, 9]  else 0
    return winter, summer, monsoon

def get_season_label(month):
    """Human-readable IMD season name, used in UI output."""
    if month in [12, 1, 2]:   return 'Winter'
    if month in [3, 4, 5]:    return 'Pre-Monsoon'
    if month in [6, 7, 8, 9]: return 'Monsoon'
    return 'Post-Monsoon'

def main():
    raw = sys.argv[1] if len(sys.argv) > 1 else "{}"
    inp = json.loads(raw)

    # Load artifacts
    model     = joblib.load(MODEL_PATH)
    with open(CITY_DATA_PATH) as f:
        city_db = json.load(f)

    city = inp.get("city", "Delhi")
    if city not in city_db:
        city = list(city_db.keys())[0]

    profile = city_db[city]
    now     = datetime.now()
    month   = now.month
    dow     = now.weekday()
    winter, summer, monsoon = get_seasonal(month)

    # Read inputs (manual or auto)
    aqi      = float(inp.get("AQI",        profile["latest"]["AQI"]))
    lag1     = float(inp.get("AQI_lag1",   profile["latest"]["AQI_lag1"]))
    lag3     = float(inp.get("AQI_lag3",   profile["latest"]["AQI_lag3"]))
    lag5     = float(inp.get("AQI_lag5",   profile["latest"]["AQI_lag5"]))
    lag7     = float(inp.get("AQI_lag7",   profile["latest"]["AQI_lag7"]))
    pm25     = float(inp.get("PM2.5",      profile["latest"]["PM2.5"]))
    pm25_l3  = float(inp.get("PM2.5_lag3", profile["latest"]["PM2.5_lag3"]))
    pm25_l7  = float(inp.get("PM2.5_lag7", profile["latest"]["PM2.5_lag7"]))
    pm10     = float(inp.get("PM10",       profile["latest"]["PM10"]))
    no2      = float(inp.get("NO2",        profile["latest"]["NO2"]))
    so2      = float(inp.get("SO2",        profile["latest"]["SO2"]))

    roll3, roll5, roll7 = compute_rolling(aqi, lag1, lag3, lag5, lag7)
    pm25_r7             = compute_pm25_roll7(pm25, pm25_l3, pm25_l7)

    row = {
        "AQI": aqi, "PM2.5": pm25, "PM10": pm10, "NO2": no2, "SO2": so2,
        "Elevation": profile["Elevation"],
        "Slope":     profile["Slope"],
        "TPI":       profile["TPI"],
        "Industry":  profile["Industry"],
        "AQI_lag1":  lag1, "AQI_lag3": lag3, "AQI_lag5": lag5, "AQI_lag7": lag7,
        "PM2.5_lag3": pm25_l3, "PM2.5_lag7": pm25_l7,
        "AQI_roll3":  roll3, "AQI_roll5": roll5, "AQI_roll7": roll7,
        "PM2.5_roll7": pm25_r7,
        "Month": month, "Day_of_week": dow,
        "Winter": winter, "Summer": summer, "Monsoon": monsoon
    }

    X = [[row[f] for f in FEATURES]]
    prediction = float(model.predict(X)[0])
    prediction = max(0, prediction)

    risk, message = classify(prediction)

    result = {
        "cases":    round(prediction, 1),
        "risk":     risk,
        "message":  message,
        "city":     city,
        "features": {
            "AQI_roll7":   round(roll7, 1),
            "AQI_roll3":   round(roll3, 1),
            "PM2.5_roll7": round(pm25_r7, 1),
            "month":       month,
            "season":      get_season_label(month)
        }
    }

    print(json.dumps(result))

if __name__ == "__main__":
    main()