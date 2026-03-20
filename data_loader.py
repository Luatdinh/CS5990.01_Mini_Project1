import os
import glob
import pandas as pd
from pandas.errors import EmptyDataError

REQUIRED_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Volume"}

def load_single_stock_file(file_path):
    try:
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {file_path}")
            return None

        df = pd.read_csv(file_path)

        if df.empty:
            print(f"Skipping blank file: {file_path}")
            return None

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            print(f"Skipping invalid file: {file_path} (missing columns: {sorted(missing)})")
            return None

        ticker = os.path.basename(file_path).split(".")[0].upper()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        df["ticker"] = ticker

        return df

    except EmptyDataError:
        print(f"Skipping unreadable empty CSV: {file_path}")
        return None
    except Exception as e:
        print(f"Skipping bad file: {file_path} ({e})")
        return None


def load_stock_directory(data_dir):
    files = glob.glob(os.path.join(data_dir, "*.txt"))
    frames = []

    for file_path in files:
        df = load_single_stock_file(file_path)
        if df is not None:
            frames.append(df)

    if not frames:
        raise ValueError("No valid stock files were loaded from data/raw")

    return pd.concat(frames, ignore_index=True)