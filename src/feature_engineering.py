"""
feature_engineering.py
=======================
FactoryGuard AI — Feature Engineering Module
Author : Member 3 (Nikhil)
Dataset : NASA CMAPSS Turbofan Engine (FD001)

Functions
---------
load_data(filepath)
add_rolling_features(df, sensors, windows)
add_lag_features(df, sensors, lags)
build_train_test_split(df, drop_cols, test_size)
build_features(filepath, save_path)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENSOR_COLS = [f"s{i}" for i in range(1, 22)]          # s1 … s21
DROP_COLS   = ["unit_nr", "time_cycles", "max_cycles", "RUL"]
WINDOWS     = [5, 10, 20]                               # rolling window sizes
LAGS        = [1, 2]                                    # lag steps


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the pre-processed CMAPSS CSV that already contains RUL and failure columns.

    Parameters
    ----------
    filepath : str
        Path to train_FD001_with_RUL.csv

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with shape (n_rows, 29).
        Columns: unit_nr, time_cycles, op_setting_1-3, s1-s21,
                 max_cycles, RUL, failure.

    Example
    -------
    >>> df = load_data("data/train_FD001_with_RUL.csv")
    >>> print(df.shape)
    (24640, 29)
    """
    df = pd.read_csv(filepath)
    print(f"[load_data] Loaded {df.shape[0]:,} rows × {df.shape[1]} cols from '{filepath}'")
    return df


# ---------------------------------------------------------------------------
# 2. Rolling features  (mean, std, EMA)
# ---------------------------------------------------------------------------

def add_rolling_features(
    df: pd.DataFrame,
    sensors: list = SENSOR_COLS,
    windows: list = WINDOWS,
) -> pd.DataFrame:
    """
    Add rolling mean, rolling std, and EMA for each sensor column.

    All rolling operations are applied **within each machine unit** using
    groupby('unit_nr') so readings from different engines never mix.

    Uses pd.concat internally to avoid the DataFrame fragmentation
    PerformanceWarning that occurs when columns are added one by one.

    Parameters
    ----------
    df      : pd.DataFrame  — input DataFrame (must contain 'unit_nr' column)
    sensors : list[str]     — sensor column names to process  (default: s1-s21)
    windows : list[int]     — rolling window sizes            (default: [5,10,20])

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new feature columns appended.
        New column count = len(sensors) × len(windows) × 3
                        = 21 × 3 × 3 = 189 columns.

    Example
    -------
    >>> df = add_rolling_features(df, sensors=SENSOR_COLS, windows=[5, 10])
    >>> print(df.shape)   # 29 original + 21*2*3 = 126 new
    (24640, 155)
    """
    new_cols = {}

    for col in sensors:
        grouped = df.groupby("unit_nr")[col]

        for w in windows:
            new_cols[f"{col}_roll_mean_{w}"] = grouped.transform(
                lambda x, w=w: x.rolling(w, min_periods=1).mean()
            )
            new_cols[f"{col}_roll_std_{w}"] = grouped.transform(
                lambda x, w=w: x.rolling(w, min_periods=1).std().fillna(0)
            )
            new_cols[f"{col}_ema_{w}"] = grouped.transform(
                lambda x, w=w: x.ewm(span=w, adjust=False).mean()
            )

    # concat once — avoids PerformanceWarning from repeated df[col] = ...
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    print(f"[add_rolling_features] Done — shape now {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 3. Lag features
# ---------------------------------------------------------------------------

def add_lag_features(
    df: pd.DataFrame,
    sensors: list = SENSOR_COLS,
    lags: list = LAGS,
) -> pd.DataFrame:
    """
    Add lag features for each sensor column within each machine unit.

    Lag features capture the sensor reading N cycles ago and help the model
    detect trends (is the sensor rising or falling?).

    Parameters
    ----------
    df      : pd.DataFrame  — input DataFrame (must contain 'unit_nr' column)
    sensors : list[str]     — sensor column names to process  (default: s1-s21)
    lags    : list[int]     — lag steps to compute            (default: [1, 2])

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new lag columns appended.
        New column count = len(sensors) × len(lags) = 21 × 2 = 42 columns.
        NaN values (first N rows per unit) are filled with 0.

    Example
    -------
    >>> df = add_lag_features(df, sensors=SENSOR_COLS, lags=[1, 2])
    >>> print(df["s1_lag1"].head())
    0    0.0
    1    518.67
    ...
    """
    new_cols = {}

    for col in sensors:
        grouped = df.groupby("unit_nr")[col]

        for lag in lags:
            new_cols[f"{col}_lag{lag}"] = (
                grouped.transform(lambda x, l=lag: x.shift(l)).fillna(0)
            )

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    print(f"[add_lag_features] Done — shape now {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 4. Train / test split  (time-aware: split by machine unit, not row shuffle)
# ---------------------------------------------------------------------------

def build_train_test_split(
    df: pd.DataFrame,
    drop_cols: list = DROP_COLS,
    test_size: float = 0.2,
):
    """
    Split the feature DataFrame into train and test sets.

    Split is done by **machine unit** (unit_nr), not by row shuffle.
    This is mandatory for time-series data — shuffling rows would leak
    future sensor readings into the training set.

    Parameters
    ----------
    df        : pd.DataFrame  — fully featured DataFrame
    drop_cols : list[str]     — columns to remove before splitting
                                (default: unit_nr, time_cycles, max_cycles, RUL)
    test_size : float         — fraction of units held out for testing (default 0.2)

    Returns
    -------
    tuple : (X_train, X_test, y_train, y_test)
        All four are pd.DataFrame / pd.Series.

    Example
    -------
    >>> X_train, X_test, y_train, y_test = build_train_test_split(df)
    >>> print(X_train.shape, X_test.shape)
    (19468, 256) (5172, 256)
    """
    units       = df["unit_nr"].unique()
    n_test      = max(1, int(len(units) * test_size))
    train_units = units[: len(units) - n_test]
    test_units  = units[len(units) - n_test :]

    train = df[df["unit_nr"].isin(train_units)]
    test  = df[df["unit_nr"].isin(test_units)]

    X_train = train.drop(columns=drop_cols)
    y_train = train["failure"]
    X_test  = test.drop(columns=drop_cols)
    y_test  = test["failure"]

    print(f"[build_train_test_split] Train units: {len(train_units)} | "
          f"Test units: {len(test_units)}")
    print(f"[build_train_test_split] X_train: {X_train.shape} | "
          f"X_test: {X_test.shape}")
    print(f"[build_train_test_split] Positive rate — "
          f"train: {y_train.mean():.3f} | test: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 5. Master pipeline — runs everything end to end
# ---------------------------------------------------------------------------

def build_features(
    filepath: str,
    save_path: str = "data/features_df.pkl",
) -> pd.DataFrame:
    """
    Full feature engineering pipeline — load → rolling → lag → save.

    Runs load_data → add_rolling_features → add_lag_features in order,
    then saves the result to disk using joblib.

    Parameters
    ----------
    filepath  : str  — path to train_FD001_with_RUL.csv
    save_path : str  — where to write features_df.pkl (default: data/features_df.pkl)

    Returns
    -------
    pd.DataFrame
        Fully featured DataFrame ready for train/test split.

    Example
    -------
    >>> df = build_features("data/train_FD001_with_RUL.csv")
    >>> print(df.shape)
    (24640, 260)
    """
    df = load_data(filepath)
    df = add_rolling_features(df, sensors=SENSOR_COLS, windows=WINDOWS)
    df = add_lag_features(df, sensors=SENSOR_COLS, lags=LAGS)

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(df, save_path, compress=3)
    print(f"[build_features] Saved features DataFrame → '{save_path}'")

    return df


# ---------------------------------------------------------------------------
# 6. Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/train_FD001_with_RUL.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/features_df.pkl"

    print("=" * 60)
    print("FactoryGuard AI — Feature Engineering")
    print("=" * 60)

    df_features = build_features(filepath=input_path, save_path=output_path)

    X_train, X_test, y_train, y_test = build_train_test_split(df_features)

    print("\nFeature engineering complete!")
    print(f"  Total features : {X_train.shape[1]}")
    print(f"  Train rows     : {len(X_train):,}")
    print(f"  Test rows      : {len(X_test):,}")
    print("=" * 60)
