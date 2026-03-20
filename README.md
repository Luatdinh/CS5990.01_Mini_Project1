# Stock Market Forecasting Project

This project trains neural-network regressors to forecast stock performance across multiple horizons using historical OHLCV data.

## What this project does

- Loads stock history files from `data/raw/*.txt`
- Creates technical features from historical prices and volume
- Trains regression models for:
  - next day (1 trading day)
  - next week (5 trading days)
  - next month (21 trading days)
  - next 2 months (42 trading days)
  - next 3 months (63 trading days)
  - next 6 months (126 trading days)
  - next year (252 trading days)
- Evaluates each model with:
  - MAE
  - RMSE
  - R²
  - directional accuracy
- Produces stock-level predictions and investment signals

## Project structure

```text
stock_market_forecasting_project/
├── data/
│   └── raw/
├── models/
├── outputs/
│   ├── metrics/
│   └── plots/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── prediction.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run training

```bash
python main.py
```

This will:

- train one neural network regressor per horizon
- save models into `models/`
- write metrics into `outputs/metrics/model_metrics.csv`
- write stock predictions into `outputs/predictions.csv`
- write plots into `outputs/plots/`

## Input data format

Each raw file should follow this structure:

```csv
Date,Open,High,Low,Close,Volume,OpenInt
1999-11-18,30.713,33.754,27.002,29.702,66277506,0
```

The included sample data uses your uploaded files:

- `a.us.txt`
- `aa.us.txt`

## Prediction target

The model predicts **future return**, not just raw price.

Example for 1-month target:

```text
future_return_1m = (Close[t+21] - Close[t]) / Close[t]
```

This is better than directly predicting price because it makes different stocks more comparable.

## Investment signal

The project converts predicted returns into a simple recommendation:

- `Buy` if predicted return >= 8%
- `Hold` if predicted return >= 2%
- `Weak` if predicted return >= -5%
- `Avoid` otherwise

You can change these thresholds inside `src/utils.py`.

## Important note

This project is a **decision-support tool**, not guaranteed financial advice. It learns from historical price patterns and does not know future news, earnings, regulations, wars, or macroeconomic shocks.

