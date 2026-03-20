from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from src.config import TRADING_HORIZONS
from src.utils import investment_signal


def _target_date_column_name(horizon_name: str) -> str:
    return f"target_{horizon_name}_date"


def add_target_dates(output: pd.DataFrame, requested_horizons: list[str]) -> pd.DataFrame:
    output = output.copy()
    base_date = pd.to_datetime(output["start_date"])

    for horizon_name in requested_horizons:
        col_name = f"pred_{horizon_name}"
        if col_name in output.columns:
            output[_target_date_column_name(horizon_name)] = base_date + BDay(TRADING_HORIZONS[horizon_name])

    date_cols = ["start_date"] + [_target_date_column_name(h) for h in requested_horizons]
    for col in date_cols:
        if col in output.columns:
            output[col] = pd.to_datetime(output[col], errors="coerce").dt.strftime("%Y-%m-%d")

    return output


def reorder_columns(output: pd.DataFrame, requested_horizons: list[str]) -> pd.DataFrame:
    desired_order = ["ticker", "start_date"]

    for horizon_name in requested_horizons:
        date_col = _target_date_column_name(horizon_name)
        if date_col in output.columns:
            desired_order.append(date_col)

    desired_order.append("latest_close")

    for horizon_name in requested_horizons:
        pred_col = f"pred_{horizon_name}"
        if pred_col in output.columns:
            desired_order.append(pred_col)

    if "signal" in output.columns:
        desired_order.append("signal")

    existing_cols = [col for col in desired_order if col in output.columns]
    remaining_cols = [col for col in output.columns if col not in existing_cols]
    return output[existing_cols + remaining_cols]


def build_signal(output: pd.DataFrame, requested_horizons: list[str]) -> pd.DataFrame:
    output = output.copy()

    # If month is requested, use month prediction for signal
    if "pred_month" in output.columns:
        output["signal"] = output["pred_month"].apply(
            lambda x: investment_signal(x) if pd.notna(x) else "Insufficient Data"
        )
        return output

    # If only one horizon is requested, use that horizon for signal
    if len(requested_horizons) == 1:
        pred_col = f"pred_{requested_horizons[0]}"
        if pred_col in output.columns:
            output["signal"] = output[pred_col].apply(
                lambda x: investment_signal(x) if pd.notna(x) else "Insufficient Data"
            )
            return output

    # Otherwise use the average of requested predictions
    pred_cols = [f"pred_{h}" for h in requested_horizons if f"pred_{h}" in output.columns]
    if pred_cols:
        output["signal_score"] = output[pred_cols].mean(axis=1, skipna=True)
        output["signal"] = output["signal_score"].apply(
            lambda x: investment_signal(x) if pd.notna(x) else "Insufficient Data"
        )
        output = output.drop(columns=["signal_score"])

    return output


def predict_latest_per_stock(
    feature_df: pd.DataFrame,
    models_dir: Path,
    requested_horizons: list[str],
) -> pd.DataFrame:
    latest_rows = feature_df.sort_values(["ticker", "Date"]).groupby("ticker").tail(1).copy()

    output = latest_rows[["ticker", "Date", "Close"]].rename(
        columns={"Date": "start_date", "Close": "latest_close"}
    ).copy()

    for horizon_name in requested_horizons:
        model_path = models_dir / f"model_{horizon_name}.joblib"
        pred_col = f"pred_{horizon_name}"

        if not model_path.exists():
            output[pred_col] = np.nan
            continue

        payload = joblib.load(model_path)
        model = payload["model"]
        scaler = payload["scaler"]
        feature_columns = payload["feature_columns"]

        clean_latest = latest_rows.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns).copy()
        X_latest = scaler.transform(clean_latest[feature_columns])
        preds = model.predict(X_latest)

        pred_map = dict(zip(clean_latest["ticker"], preds))
        output[pred_col] = output["ticker"].map(pred_map)

    output = add_target_dates(output, requested_horizons)
    output = build_signal(output, requested_horizons)
    output = reorder_columns(output, requested_horizons)

    sort_col = f"pred_{requested_horizons[0]}" if len(requested_horizons) == 1 else (
        "pred_month" if "month" in requested_horizons and "pred_month" in output.columns else "ticker"
    )
    ascending = False if sort_col != "ticker" else True

    return output.sort_values(sort_col, ascending=ascending, na_position="last").reset_index(drop=True)