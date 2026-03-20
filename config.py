from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"
BUY_THRESHOLD = 0.08
HOLD_THRESHOLD = 0.02
WEAK_THRESHOLD = -0.05

TRADING_HORIZONS = {
    "day": 1,
    "week": 5,
    "month": 21,
    "2month": 42,
    "3month": 63,
    "6month": 126,
    "year": 252,
}

RAW_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"]
SEED = 42
