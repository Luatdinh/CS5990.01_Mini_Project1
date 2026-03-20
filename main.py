from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import pandas as pd

from src.config import DATA_DIR, METRICS_DIR, MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, TRADING_HORIZONS
from src.data_loader import load_stock_directory
from src.feature_engineering import add_features, add_targets, get_feature_columns
from src.modeling import time_split, train_single_horizon
from src.prediction import predict_latest_per_stock
from src.utils import ensure_directories


HORIZON_ALIASES = {
    "day": "day",
    "1day": "day",
    "1-day": "day",
    "d": "day",

    "week": "week",
    "1week": "week",
    "1-week": "week",
    "w": "week",

    "month": "month",
    "1month": "month",
    "1-month": "month",
    "m": "month",

    "2month": "2month",
    "2months": "2month",
    "2-month": "2month",

    "3month": "3month",
    "3months": "3month",
    "3-month": "3month",

    "6month": "6month",
    "6months": "6month",
    "6-month": "6month",

    "year": "year",
    "1year": "year",
    "1-year": "year",
    "y": "year",
}


def normalize_horizons(raw_horizons: list[str]) -> list[str]:
    normalized = []
    for item in raw_horizons:
        key = item.strip().lower().replace(" ", "")
        mapped = HORIZON_ALIASES.get(key)
        if mapped is None:
            raise ValueError(
                f"Invalid horizon '{item}'. Valid examples: day, week, month, year, 1day, 1week, 1month, 1year"
            )
        if mapped not in normalized:
            normalized.append(mapped)
    return normalized


def parse_args():
    parser = argparse.ArgumentParser(description="Train stock forecasting neural-network regressors.")
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=["day", "week", "month"],
        help="Examples: day week month year  OR  month  OR  1day  OR  1week  OR  1month  OR  1year",
    )
    return parser.parse_args()


def save_prediction_plot(pred_df: pd.DataFrame, plot_path, requested_horizons: list[str]):
    if pred_df.empty:
        return

    plot_horizon = "month" if "month" in requested_horizons else requested_horizons[0]
    pred_col = f"pred_{plot_horizon}"

    if pred_col not in pred_df.columns:
        return

    plot_df = pred_df.dropna(subset=[pred_col]).copy()
    if plot_df.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.bar(plot_df["ticker"], plot_df[pred_col])
    plt.title(f"Predicted {plot_horizon.title()} Return by Ticker")
    plt.xlabel("Ticker")
    plt.ylabel("Predicted Return")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def main() -> None:
    args = parse_args()
    requested_horizons = normalize_horizons(args.horizons)

    ensure_directories([MODELS_DIR, OUTPUTS_DIR, METRICS_DIR, PLOTS_DIR])

    print("Loading stock files...")
    raw_df = load_stock_directory(DATA_DIR)
    print(f"Loaded {len(raw_df):,} rows across {raw_df['ticker'].nunique()} tickers.")

    print("Building features and targets...")
    feature_df = add_features(raw_df)
    modeling_df = add_targets(feature_df)
    feature_columns = get_feature_columns()

    print("Splitting dataset by time...")
    train_df, valid_df, test_df = time_split(modeling_df)

    metrics_records = []
    for horizon_name in requested_horizons:
        target_col = f"target_{horizon_name}"
        print(f"Training model for {horizon_name} horizon...")
        metrics = train_single_horizon(
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            feature_cols=feature_columns,
            target_col=target_col,
            model_output_path=MODELS_DIR / f"model_{horizon_name}.joblib",
        )
        metrics_records.append(metrics)

    metrics_df = pd.DataFrame(metrics_records)
    metrics_path = METRICS_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")

    predictions_df = predict_latest_per_stock(feature_df, MODELS_DIR, requested_horizons)
    predictions_path = OUTPUTS_DIR / "predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved latest predictions to {predictions_path}")

    plot_path = PLOTS_DIR / "predicted_returns.png"
    save_prediction_plot(predictions_df, plot_path, requested_horizons)
    print(f"Saved plot to {plot_path}")

    print("\nTop predictions:")
    print(predictions_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()