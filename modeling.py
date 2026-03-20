from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.config import SEED
from src.utils import safe_directional_accuracy


def time_split(df: pd.DataFrame, train_ratio: float = 0.7, valid_ratio: float = 0.15):
    unique_dates = sorted(pd.to_datetime(df["Date"]).dropna().unique())
    n_dates = len(unique_dates)

    train_end_idx = int(n_dates * train_ratio)
    valid_end_idx = int(n_dates * (train_ratio + valid_ratio))

    train_dates = unique_dates[:train_end_idx]
    valid_dates = unique_dates[train_end_idx:valid_end_idx]
    test_dates = unique_dates[valid_end_idx:]

    train_df = df[df["Date"].isin(train_dates)].copy()
    valid_df = df[df["Date"].isin(valid_dates)].copy()
    test_df = df[df["Date"].isin(test_dates)].copy()

    return train_df, valid_df, test_df


def build_regressor() -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        learning_rate_init=0.001,
        batch_size=512,
        max_iter=40,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=SEED,
    )


def prepare_xy(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    clean_df = df[feature_cols + [target_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = clean_df[feature_cols]
    y = clean_df[target_col]
    return X, y


def train_single_horizon(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_output_path: Path,
) -> dict:
    X_train, y_train = prepare_xy(train_df, feature_cols, target_col)
    X_valid, y_valid = prepare_xy(valid_df, feature_cols, target_col)
    X_test, y_test = prepare_xy(test_df, feature_cols, target_col)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    model = build_regressor()
    model.fit(X_train_scaled, y_train)

    valid_pred = model.predict(X_valid_scaled)
    test_pred = model.predict(X_test_scaled)

    metrics = {
        "target": target_col,
        "train_rows": len(X_train),
        "valid_rows": len(X_valid),
        "test_rows": len(X_test),
        "valid_mae": float(mean_absolute_error(y_valid, valid_pred)),
        "valid_rmse": float(np.sqrt(mean_squared_error(y_valid, valid_pred))),
        "valid_r2": float(r2_score(y_valid, valid_pred)),
        "valid_directional_accuracy": safe_directional_accuracy(y_valid, pd.Series(valid_pred, index=y_valid.index)),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "test_r2": float(r2_score(y_test, test_pred)),
        "test_directional_accuracy": safe_directional_accuracy(y_test, pd.Series(test_pred, index=y_test.index)),
    }

    payload = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_cols,
        "target_column": target_col,
    }
    joblib.dump(payload, model_output_path)

    return metrics
