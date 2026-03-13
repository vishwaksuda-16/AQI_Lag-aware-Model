"""
train_model.py
Trains a HistGradientBoostingRegressor on your real dataset.

HOW TO USE:
  Just paste your dataset path below where it says DATASET_PATH.
  Then run:  python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib
import json
import os

# Always resolve paths relative to this script file, not the working directory
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(SCRIPT_DIR, "..", "models")

np.random.seed(42)

# ============================================================
#   PASTE YOUR DATASET PATH HERE  ↓
# ============================================================
DATASET_PATH = "D:\Projects\AQI\hospital-demand-predictor\datasets\Model_Ready_Dataset.csv"
# ============================================================
# Examples:
#   Windows : r"C:\Users\John\Desktop\Model_Ready_Dataset.csv"
#   Mac/Linux: "/home/john/datasets/Model_Ready_Dataset.csv"
#   Same folder as this script: "Model_Ready_Dataset.csv"
# ============================================================

# ── Load dataset ─────────────────────────────────────────────────────────────
print(f"\nLoading dataset from: {DATASET_PATH}")

if not os.path.exists(DATASET_PATH):
    print(f"\n ERROR: File not found at the path you provided.")
    print(f"  Please check the path and try again.")
    print(f"  Path given: {DATASET_PATH}\n")
    exit(1)

df = pd.read_csv(DATASET_PATH)
df['Date'] = pd.to_datetime(df['Date'])

print(f" Loaded successfully!")
print(f" Shape     : {df.shape}")
print(f" Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f" Cities    : {df['City'].nunique()} cities")
print(f" Columns   : {list(df.columns)}\n")

# ── Feature list (must match predict.py) ─────────────────────────────────────
FEATURES = [
    "AQI", "PM2.5", "PM10", "NO2", "SO2",
    "Elevation", "Slope", "TPI", "Industry",
    "AQI_lag1", "AQI_lag3", "AQI_lag5", "AQI_lag7",
    "PM2.5_lag3", "PM2.5_lag7",
    "AQI_roll3", "AQI_roll5", "AQI_roll7", "PM2.5_roll7",
    "Month", "Day_of_week", "Winter", "Summer", "Monsoon"
]

train_df = df[df["Date"].dt.year <= 2023]
test_df  = df[df["Date"].dt.year >= 2024]

X_train = train_df[FEATURES]
y_train = train_df["Cases_t_plus3"]

best_gbr = HistGradientBoostingRegressor(
    learning_rate=0.1,
    max_depth=12,
    max_iter=200,
    random_state=42
)
best_gbr.fit(X_train, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
X_test = test_df[FEATURES]
y_test = test_df["Cases_t_plus3"]
preds  = best_gbr.predict(X_test)
mae    = mean_absolute_error(y_test, preds)
rmse   = np.sqrt(mean_squared_error(y_test, preds))
r2     = r2_score(y_test, preds)
print(f"Test MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")

# ── Save model ────────────────────────────────────────────────────────────────
os.makedirs(MODELS_DIR, exist_ok=True)
model_path = os.path.join(MODELS_DIR, "hospital_model.pkl")
joblib.dump(best_gbr, model_path)
print(f"Saved: {model_path}")

# ── Save city profiles & latest pollution data (read from real dataset) ───────
print("\nBuilding city profiles from dataset...")
city_data = {}

for city in df["City"].unique():
    city_rows = df[df["City"] == city].sort_values("Date")
    last = city_rows.iloc[-1]

    # Read geo/industry directly from the dataset (they are constant per city)
    city_data[city] = {
        "Elevation": round(float(city_rows["Elevation"].iloc[0]), 4),
        "Slope":     round(float(city_rows["Slope"].iloc[0]),     4),
        "TPI":       round(float(city_rows["TPI"].iloc[0]),       4),
        "Industry":  round(float(city_rows["Industry"].iloc[0]),  4),
        "latest": {
            "AQI":        round(float(last["AQI"]),         1),
            "AQI_lag1":   round(float(last["AQI_lag1"]),    1),
            "AQI_lag3":   round(float(last["AQI_lag3"]),    1),
            "AQI_lag5":   round(float(last["AQI_lag5"]),    1),
            "AQI_lag7":   round(float(last["AQI_lag7"]),    1),
            "PM2.5":      round(float(last["PM2.5"]),       1),
            "PM2.5_lag3": round(float(last["PM2.5_lag3"]),  1),
            "PM2.5_lag7": round(float(last["PM2.5_lag7"]),  1),
            "PM10":       round(float(last["PM10"]),        1),
            "NO2":        round(float(last["NO2"]),         1),
            "SO2":        round(float(last["SO2"]),         1),
        }
    }

city_json_path = os.path.join(MODELS_DIR, "city_data.json")
with open(city_json_path, "w") as f:
    json.dump(city_data, f, indent=2)

print(f"Saved: {city_json_path}  ({len(city_data)} cities)")
print("\n Training complete! Your model is ready.")
print(" You can now run:  node app.js")