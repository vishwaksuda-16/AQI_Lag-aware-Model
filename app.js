const express = require('express');
const path    = require('path');
const routes  = require('./routes/predict');

const app  = express();
const PORT = process.env.PORT || 3000;

app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use('/', routes);

app.listen(PORT, () => {
  console.log(`Hospital Demand Predictor running at http://localhost:${PORT}`);
});
