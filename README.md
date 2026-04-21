# AQI → Health: Lag-Aware Respiratory Hospital Demand Forecasting

> **Study Title:** Lag-Aware and Topography-Informed Air Quality Exposure Modeling for Urban Hospital Demand Forecasting  
> **Deployment:** Node.js + Express + Python (scikit-learn) + Open-Meteo Weather API

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [How to Run](#2-how-to-run)
3. [UI Architecture & Real-Time Data](#3-ui-architecture--real-time-data)
4. [How Real-Time Data Works](#4-how-real-time-data-works)
5. [Dataset Specifics (Phase 1 & 2)](#5-dataset-specifics-phase-1--2)
6. [Phase 4 — Machine Learning Modeling](#6-phase-4--machine-learning-modeling)
7. [Deployment Architecture](#7-deployment-architecture)
8. [Author's Interpretation](#8-authors-interpretation)

---

## 1. Project Overview

**Problem Statement:** Existing models ignore the delayed biological impact of pollution and the role of terrain (elevation/slope) in pollutant concentration.

**Key Innovation:** Combining 7-day rolling exposure metrics with topographic position indices (TPI) to forecast respiratory hospital admissions 3 days ahead.

**Core Research Hypothesis:** Does cumulative pollution exposure over days affect respiratory health more than the same-day pollution level?  
**Answer:** Yes — the lag-aware model outperforms the same-day model by **7.45% MAE**.

---

## 2. How to Run

```bash
# Install dependencies
npm install

# Start server (requires Python with scikit-learn installed)
node app.js

# Open in browser
http://localhost:3000
```

**Python requirements:**
```bash
pip install scikit-learn joblib numpy pandas
```

---

## 3. UI Architecture & Real-Time Data

The web application is built with **Node.js + Express + EJS** for server-side rendering.  
All real-time interactivity is handled in the browser via **vanilla JavaScript** — no frontend framework is needed.

### UI Layout

```
┌──────────────────────────────────────────────────────────────┐
│  HEADER: Navigation tabs (Forecast | Model Insights)         │
├──────────────────────────────────────────────────────────────┤
│  STATUS STRIP (7 live cells)                                 │
│  Clock | AQI Gauge | Live Weather | Pollutants | Trend | Risk│
├───────────────────────────────┬──────────────────────────────┤
│  POLLUTION INPUT FORM         │  WEATHER WIDGET              │
│  (auto-fills on city change)  │  (Open-Meteo live API)       │
│                               ├──────────────────────────────┤
│                               │  FORECAST OUTPUT             │
│                               ├──────────────────────────────┤
│                               │  CITY GEO PROFILE            │
└───────────────────────────────┴──────────────────────────────┘
```

### Model Insights Tab

A separate tab (`#insights`) shows:
- 4 KPI cards (MAE, Lag improvement %, Pollution dominance %, CV stability)
- 7 Chart.js charts covering all Phase 4 results
- Algorithm benchmark table
- City vulnerability ranking
- Training pipeline timeline

---

## 4. How Real-Time Data Works

### 4.1 City Pollution Data (from Trained Dataset)

When the server starts, `routes/predict.js` loads `models/city_data.json` — a pre-computed JSON file extracted from the final training dataset (`Model_Ready_Dataset.csv`).

```js
// routes/predict.js
const CITY_DATA_PATH = path.join(__dirname, '../models/city_data.json');
function getCityData() {
  return JSON.parse(fs.readFileSync(CITY_DATA_PATH, 'utf8'));
}
```

This is injected into the EJS template:

```js
// The server sends all city data to the browser on page load:
res.render('index', {
  cities: Object.keys(cityData).sort(),
  cityDataJSON: JSON.stringify(cityData),  // ← serialized to JS variable
  result: null,
  error: null
});
```

In the browser, this becomes:

```js
var CITY_DATA = <%- cityDataJSON %>;  // EJS unescaped injection
```

Each city entry in `CITY_DATA` contains:

```json
{
  "Agartala": {
    "latest": {
      "AQI": 40.2,
      "AQI_lag1": 38.5,
      "AQI_lag3": 35.1,
      "AQI_lag5": 33.4,
      "AQI_lag7": 31.2,
      "PM2.5": 22.1,
      "PM2.5_lag3": 20.5,
      "PM2.5_lag7": 19.0,
      "PM10": 45.6,
      "NO2": 18.3,
      "SO2": 8.4
    },
    "Elevation": 16.67,
    "Slope": 0.24,
    "TPI": -2.67,
    "Industry": 4.45
  }
}
```

### 4.2 Auto-Fill on City Selection

When a city is selected, **all 11 input fields are automatically populated** with the city's latest dataset values, with a green flash animation:

```js
var FORM_MAP = {
  'AQI':       'AQI',
  'AQI_lag1':  'AQI_lag1',
  'AQI_lag3':  'AQI_lag3',
  'AQI_lag5':  'AQI_lag5',
  'AQI_lag7':  'AQI_lag7',
  'PM25':      'PM2.5',
  'PM25_lag3': 'PM2.5_lag3',
  'PM25_lag7': 'PM2.5_lag7',
  'PM10':      'PM10',
  'NO2':       'NO2',
  'SO2':       'SO2'
};

function fillAllFields(city) {
  var l = CITY_DATA[city].latest;
  Object.keys(FORM_MAP).forEach(function(fieldId) {
    var el = document.getElementById(fieldId);
    el.value = parseFloat(l[FORM_MAP[fieldId]]).toFixed(1);
    el.classList.add('autofilled');           // green glow
    setTimeout(() => el.classList.remove('autofilled'), 2000);
  });
}
```

Fields remain **fully editable** after auto-fill, allowing custom scenario testing.

### 4.3 Live Weather via Open-Meteo API

**Open-Meteo** is a free, open-source weather API that requires **no API key**.  
It returns real-time atmospheric data using latitude/longitude coordinates.

**API Endpoint used:**
```
GET https://api.open-meteo.com/v1/forecast
  ?latitude={lat}
  &longitude={lon}
  &current=temperature_2m,relative_humidity_2m,apparent_temperature,
           wind_speed_10m,weather_code,precipitation,uv_index
  &timezone=auto
```

**City coordinates are hardcoded** in the browser JS:

```js
var CITY_COORDS = {
  'Agartala':         [23.8315, 91.2868],
  'Delhi':            [28.6139, 77.2090],
  'Mandi Gobindgarh': [30.6711, 76.3083],
  'Tiruppur':         [11.1085, 77.3411],
  // ... all 20 cities
};
```

**The fetch is done client-side** (CORS is allowed by Open-Meteo):

```js
async function fetchWeather(city) {
  var coords = CITY_COORDS[city];
  var url = `https://api.open-meteo.com/v1/forecast`
    + `?latitude=${coords[0]}&longitude=${coords[1]}`
    + `&current=temperature_2m,relative_humidity_2m,apparent_temperature,`
    + `wind_speed_10m,weather_code,precipitation,uv_index&timezone=auto`;

  var response = await fetch(url);
  var data = await response.json();
  renderWeatherWidget(city, data.current);
}
```

**WMO Weather codes are decoded** to icons and descriptions:

```js
var WMO_MAP = {
  0:  ['☀️',  'Clear Sky'],
  2:  ['⛅', 'Partly Cloudy'],
  61: ['🌧', 'Light Rain'],
  95: ['⛈',  'Thunderstorm'],
  // ...
};
```

**Weather is displayed in two places:**
1. **Status strip** — temperature, icon, humidity, wind speed, feels like
2. **Weather widget panel** — full card with temperature, condition, feels like, humidity, wind, UV index, precipitation, coordinates, update time

### 4.4 AQI Gauge (Canvas Arc)

The AQI is visualised as a semicircular arc gauge drawn on an HTML5 `<canvas>` element:

```js
function drawAQIArc(aqi) {
  var color = aqi<=100 ? '#22c55e' : aqi<=200 ? '#f59e0b' : '#ef4444';
  var pct   = Math.min(aqi / 500, 1);
  // Draw background arc
  ctx.arc(cx, cy, r, startAngle, endAngle);
  ctx.strokeStyle = '#1e2a3d'; ctx.stroke();
  // Draw value arc
  ctx.arc(cx, cy, r, startAngle, startAngle + pct*(endAngle-startAngle));
  ctx.strokeStyle = color; ctx.stroke();
}
```

Color thresholds follow CPCB (India) AQI standards:
| AQI Range | Color | Category |
|-----------|-------|----------|
| 0–100 | 🟢 Green | Good / Satisfactory |
| 101–200 | 🟡 Amber | Moderate / Poor |
| 201–500 | 🔴 Red | Very Poor / Severe |

### 4.5 7-Day Trend Sparkline

A mini Chart.js line chart is drawn from the city's lag fields (AQI_lag7 through AQI today), showing how pollution changed over the past week:

```js
drawMiniTrend([l7, interpolated, l5, interpolated, l3, interpolated, l1, aqi]);
```

Each point is colored green/amber/red by its AQI value.

### 4.6 Pollutant Bars

Four horizontal progress bars (PM₂.₅, PM₁₀, NO₂, SO₂) are normalised against safe maximum expected values:

```js
PM₂.₅  → max 250 µg/m³
PM₁₀   → max 350 µg/m³
NO₂    → max 200 µg/m³
SO₂    → max 100 µg/m³
```

### 4.7 Live Clock

A `setInterval` tick runs every second updating the time display:

```js
setInterval(tickClock, 1000);
```

---

## 5. Dataset Specifics (Phase 1 & 2)

| Property | Value |
|----------|-------|
| **Sources** | Multi-city India AQI data (2020–2025) + synthetic respiratory admission counts |
| **Final Shape** | 374,646 rows × 29 columns |
| **Cities** | 20 Indian urban centres |

**Feature Engineering:**

| Feature Type | Examples | Purpose |
|---|---|---|
| Lag features | `AQI_lag1` → `AQI_lag7` | Captures delayed health triggers |
| Rolling mean | `AQI_roll3`, `AQI_roll7` | Captures chronic cumulative exposure |
| Topography | Elevation, Slope, TPI | Identifies pollution-trapping basins |
| Industry | Industry Score | Proxy for local emission density |
| Season | season (1–4) | Captures weather-driven variation |

---

## 6. Phase 4 — Machine Learning Modeling

### Step 1 — Algorithm Selection

Three regression algorithms were benchmarked using a time-based split:
- **Training:** 2020–2023
- **Testing:** 2024–2025

| Algorithm | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 4.22 | 5.37 | 0.13 |
| Random Forest | 3.55 | 4.77 | 0.32 |
| **Gradient Boosting** ✓ | **3.50** | **4.70** | **0.33** |

**Gradient Boosting selected** — captures nonlinear pollution-health relationships.

### Step 2 — Same-Day vs Lag-Aware

| Model | MAE | Improvement |
|---|---|---|
| Same-Day Model | 3.79 | baseline |
| **Lag-Aware Model** | **3.51** | **+7.45%** |

**Conclusion:** Cumulative exposure affects respiratory health more than same-day levels.

### Step 3 — Forecast Horizon

| Horizon | MAE | RMSE | R² |
|---|---|---|---|
| **3-Day** ✓ | **3.50** | **4.70** | **0.34** |
| 5-Day | 3.63 | 4.84 | 0.30 |

**3-day forecasting adopted** as the primary horizon.

### Step 4 — Hyperparameter Tuning (GridSearchCV)

| Parameter | Best Value |
|---|---|
| `learning_rate` | 0.1 |
| `max_depth` | 12 |
| `max_iter` | 200 |

| Metric | Before Tuning | After Tuning |
|---|---|---|
| MAE | 3.5087 | **3.4998** |
| RMSE | 4.7036 | **4.7007** |
| R² | 0.337 | **0.338** |

### Step 5 — Time Series Cross-Validation

`TimeSeriesSplit` with 5 folds:

| Fold | MAE |
|---|---|
| 1 | 4.31 |
| 2 | 4.25 |
| 3 | 4.14 |
| 4 | 3.98 |
| 5 | 3.87 |
| **Average** | **4.11** |
| **Std Dev (σ)** | **0.16** |

Low σ = model stable across time periods.

### Step 6 — Feature Importance

| Feature Group | Importance |
|---|---|
| Rolling Exposure (7-day) | 35% |
| Rolling Exposure (3-day) | 15% |
| Rolling Exposure (5-day) | 10% |
| AQI Same-day | 7% |
| Seasonal Index | 12% |
| Elevation | 7% |
| Others | 14% |

### Step 7 — Domain Importance

| Domain | % Importance |
|---|---|
| Pollution | 72.6% |
| Season | 20.0% |
| Topography | 5.0% |
| Industry | 2.4% |

### Step 8 — City Vulnerability

Cities with **highest prediction error** (most sensitive to pollution-health dynamics):

| Rank | City | Error Magnitude |
|---|---|---|
| 1 | Tiruppur | Highest |
| 2 | Nandesari | — |
| 3 | Pathardih | — |
| 4 | Mandi Gobindgarh | — |
| 5 | Byrnihat | — |

---

## 7. Deployment Architecture

```
Browser (EJS rendered by Express)
  │
  ├──── city_data.json (loaded at boot, injected into template)
  │         └── All 20 cities: latest AQI, lag values, geo features
  │
  ├──── Open-Meteo API (called client-side, no key required)
  │         └── Live temperature, humidity, wind, UV, weather code
  │
  └──── POST /predict → Node.js → Python predict.py
            │
            └── loads hospital_model.pkl (HistGBR)
                computes rolling/lag features from input
                returns predicted cases + risk level
```

**Folder Structure:**
```
AQI---Lag-aware-Model-Hospital-Prediction/
├── app.js                  # Express server entry point
├── routes/
│   └── predict.js          # GET / and POST /predict handlers
├── views/
│   └── index.ejs           # Full UI — forecast + model insights
├── models/
│   ├── hospital_model.pkl  # Trained HistGBR model (769 KB)
│   └── city_data.json      # Pre-computed city profiles (121 KB)
├── python/
│   └── predict.py          # Python ML inference script
└── package.json
```

**Prediction Flow:**
1. User selects city → form auto-fills from `city_data.json`
2. Weather widget fetches live data from `api.open-meteo.com`
3. User clicks **Generate 3-Day Forecast**
4. Form POSTs to `/predict`
5. Node.js spawns `python predict.py '{...json...}'`
6. Python loads `hospital_model.pkl`, builds feature vector, returns JSON
7. Node.js renders EJS with result → page shows admission count + risk

**Risk Classification:**
```
Cases < 10  → LOW    (green)
Cases 10–25 → MEDIUM (amber)
Cases > 25  → HIGH   (red)
```

---

## 8. Author's Interpretation

> "I noticed that the 3-day forecast is significantly more stable than the 5-day one because hospital capacity planning usually occurs on a 72-hour cycle. This also suggests that lag-aware exposure features better match the real biological delay between pollution exposure and the onset of respiratory symptoms serious enough to require hospital admission.  
>  
> The dominance of rolling 7-day AQI (35% importance) over same-day AQI (7% importance) validates the core hypothesis: the body's response to air pollution is **cumulative and delayed**, not immediate. This has important implications for public health policy — early warnings should be based on weekly exposure trends, not just today's AQI reading."

---

## References

1. *Semi-Supervised Air Quality Forecasting via Self-Supervised Hierarchical Graph Neural Network* — IEEE, 2022
2. *The Lagged Effects of Particulate Matter on Admissions for Respiratory Diseases* — Environmental Health Perspectives
3. [Open-Meteo Weather API](https://open-meteo.com/) — Free, open-source weather API (no key required)
4. [scikit-learn HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
5. CPCB (Central Pollution Control Board) India — AQI Standards & Breakpoints
