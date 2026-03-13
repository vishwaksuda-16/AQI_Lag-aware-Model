const express = require('express');
const router  = express.Router();
const { spawn } = require('child_process');
const path = require('path');
const fs   = require('fs');

const CITY_DATA_PATH = path.join(__dirname, '../models/city_data.json');
const PREDICT_SCRIPT = path.join(__dirname, '../python/predict.py');

function getCityData() {
  return JSON.parse(fs.readFileSync(CITY_DATA_PATH, 'utf8'));
}

// ── GET / → render main page ─────────────────────────────────────────────────
router.get('/', (req, res) => {
  const cityData = getCityData();
  res.render('index', {
    cities: Object.keys(cityData).sort(),
    cityDataJSON: JSON.stringify(cityData),
    result: null,
    error: null
  });
});

// ── POST /predict → run Python, return result ─────────────────────────────────
router.post('/predict', (req, res) => {
  const cityData = getCityData();
  const body = req.body;

  const payload = {
    city:        body.city || 'Delhi',
    AQI:         parseFloat(body.AQI)        || null,
    AQI_lag1:    parseFloat(body.AQI_lag1)   || null,
    AQI_lag3:    parseFloat(body.AQI_lag3)   || null,
    AQI_lag5:    parseFloat(body.AQI_lag5)   || null,
    AQI_lag7:    parseFloat(body.AQI_lag7)   || null,
    'PM2.5':     parseFloat(body.PM25)       || null,
    'PM2.5_lag3':parseFloat(body.PM25_lag3)  || null,
    'PM2.5_lag7':parseFloat(body.PM25_lag7)  || null,
    PM10:        parseFloat(body.PM10)        || null,
    NO2:         parseFloat(body.NO2)         || null,
    SO2:         parseFloat(body.SO2)         || null,
  };

  // Remove nulls so predict.py uses city defaults for those fields
  Object.keys(payload).forEach(k => payload[k] === null && delete payload[k]);

  const args = JSON.stringify(payload);

  const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
  const py = spawn(pythonCmd, [PREDICT_SCRIPT, args]);

  let stdout = '';
  let stderr = '';

  py.stdout.on('data', d => stdout += d.toString());
  py.stderr.on('data', d => stderr += d.toString());

  py.on('close', (code) => {
    // Extract JSON from stdout (warnings may prefix it)
    const jsonMatch = stdout.match(/\{[\s\S]*\}/);

    if (jsonMatch) {
      try {
        const result = JSON.parse(jsonMatch[0]);
        return res.render('index', {
          cities: Object.keys(cityData).sort(),
          cityDataJSON: JSON.stringify(cityData),
          result,
          error: null,
          formData: body
        });
      } catch (e) {
        return res.render('index', {
          cities: Object.keys(cityData).sort(),
          cityDataJSON: JSON.stringify(cityData),
          result: null,
          error: 'Failed to parse model output: ' + e.message,
          formData: body
        });
      }
    }

    return res.render('index', {
      cities: Object.keys(cityData).sort(),
      cityDataJSON: JSON.stringify(cityData),
      result: null,
      error: 'Prediction failed. ' + (stderr || 'Unknown error'),
      formData: body
    });
  });
});

// ── GET /api/city/:name → JSON city profile (for JS auto-fill) ────────────────
router.get('/api/city/:name', (req, res) => {
  const cityData = getCityData();
  const city = req.params.name;
  if (cityData[city]) {
    res.json(cityData[city]);
  } else {
    res.status(404).json({ error: 'City not found' });
  }
});

module.exports = router;
