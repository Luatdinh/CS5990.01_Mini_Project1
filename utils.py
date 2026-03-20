from __future__ import annotations

from pathlib import Path
import pandas as pd


def ensure_directories(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def investment_signal(predicted_return: float) -> str:
    if predicted_return >= 0.08:
        return "Buy"
    if predicted_return >= 0.02:
        return "Hold"
    if predicted_return >= -0.05:
        return "Weak"
    return "Avoid"


def safe_directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(((y_true > 0) == (y_pred > 0)).mean())
