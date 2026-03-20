from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TRADING_HORIZONS


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def add_features(stock_df: pd.DataFrame) -> pd.DataFrame:
    df = stock_df.copy()

    grouped = df.groupby("ticker", group_keys=False)

    df["return_1d"] = grouped["Close"].pct_change(1)
    df["return_5d"] = grouped["Close"].pct_change(5)
    df["return_20d"] = grouped["Close"].pct_change(20)
    df["return_60d"] = grouped["Close"].pct_change(60)

    for window in [5, 10, 20, 50, 100, 200]:
        df[f"ma_{window}"] = grouped["Close"].transform(lambda s: s.rolling(window).mean())
        df[f"volatility_{window}"] = grouped["return_1d"].transform(lambda s: s.rolling(window).std())

    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)
    df["co_range"] = (df["Close"] - df["Open"]) / df["Open"].replace(0, np.nan)
    df["oc_gap"] = grouped["Open"].pct_change(1)
    df["volume_change"] = grouped["Volume"].pct_change(1)
    df["volume_ma_20"] = grouped["Volume"].transform(lambda s: s.rolling(20).mean())
    df["volume_vs_ma20"] = df["Volume"] / df["volume_ma_20"].replace(0, np.nan) - 1

    for window in [5, 10, 20, 50, 100, 200]:
        df[f"close_vs_ma_{window}"] = df["Close"] / df[f"ma_{window}"].replace(0, np.nan) - 1

    df["momentum_10"] = grouped["Close"].pct_change(10)
    df["momentum_20"] = grouped["Close"].pct_change(20)
    df["momentum_60"] = grouped["Close"].pct_change(60)

    df["rsi_14"] = grouped["Close"].transform(lambda s: compute_rsi(s, 14))

    macd_frames = []
    for ticker, sub_df in df.groupby("ticker"):
        macd, signal, hist = compute_macd(sub_df["Close"])
        temp = pd.DataFrame(
            {
                "ticker": ticker,
                "Date": sub_df["Date"].values,
                "macd": macd.values,
                "macd_signal": signal.values,
                "macd_hist": hist.values,
            }
        )
        macd_frames.append(temp)

    macd_df = pd.concat(macd_frames, ignore_index=True)
    df = df.merge(macd_df, on=["ticker", "Date"], how="left")

    rolling_mean_20 = grouped["Close"].transform(lambda s: s.rolling(20).mean())
    rolling_std_20 = grouped["Close"].transform(lambda s: s.rolling(20).std())
    upper_band = rolling_mean_20 + 2 * rolling_std_20
    lower_band = rolling_mean_20 - 2 * rolling_std_20
    df["bollinger_pos"] = (df["Close"] - lower_band) / (upper_band - lower_band)

    return df


def add_targets(feature_df: pd.DataFrame) -> pd.DataFrame:
    df = feature_df.copy()

    grouped = df.groupby("ticker", group_keys=False)
    for name, horizon in TRADING_HORIZONS.items():
        future_close = grouped["Close"].shift(-horizon)
        df[f"target_{name}"] = (future_close - df["Close"]) / df["Close"]

    return df


def get_feature_columns() -> list[str]:
    return [
        "Open", "High", "Low", "Close", "Volume",
        "return_1d", "return_5d", "return_20d", "return_60d",
        "ma_5", "ma_10", "ma_20", "ma_50", "ma_100", "ma_200",
        "volatility_5", "volatility_10", "volatility_20", "volatility_50", "volatility_100", "volatility_200",
        "hl_range", "co_range", "oc_gap", "volume_change", "volume_ma_20", "volume_vs_ma20",
        "close_vs_ma_5", "close_vs_ma_10", "close_vs_ma_20", "close_vs_ma_50", "close_vs_ma_100", "close_vs_ma_200",
        "momentum_10", "momentum_20", "momentum_60",
        "rsi_14", "macd", "macd_signal", "macd_hist", "bollinger_pos",
    ]
