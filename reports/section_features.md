# Feature Engineering

**Author:** Member 3 — Nikhil  
**Dataset:** NASA CMAPSS Turbofan Engine Degradation Dataset (FD001)  
**Script:** `src/feature_engineering.py`

---

## 1. Overview

Raw sensor readings from the CMAPSS dataset capture only the instantaneous state
of a turbofan engine at a single cycle. A single snapshot is insufficient for
predicting imminent failure because degradation is a gradual, time-dependent
process — the trend of a sensor over the last 10 or 20 cycles carries far more
predictive signal than its value at any one moment.

To capture these temporal patterns, three categories of features were engineered:
rolling statistics (mean, standard deviation, and exponential moving average) and
lag features. All operations were applied **within each machine unit** using
`groupby("unit_nr")` to ensure that sensor readings from different engines never
mix — a critical correctness requirement for multi-unit time-series data.

---

## 2. Input Data

The input file `data/train_FD001_with_RUL.csv` was loaded using `load_data()`.  
It contains **24,640 rows × 29 columns**, covering:

- `unit_nr` — machine unit identifier (100 unique engines)
- `time_cycles` — operating cycle number per engine
- `op_setting_1`, `op_setting_2`, `op_setting_3` — operational condition settings
- `s1` through `s21` — 21 continuous sensor measurements
- `max_cycles`, `RUL` — derived label columns (Remaining Useful Life)
- `failure` — binary target: 1 if RUL ≤ 30 cycles, else 0

---

## 3. Rolling Statistics Features

Rolling features were computed for all **21 sensor columns** (`s1`–`s21`) across
three window sizes: **5, 10, and 20 cycles**. For each sensor–window combination,
three statistics were produced:

| Statistic | Description | Purpose |
|-----------|-------------|---------|
| Rolling Mean | Average sensor value over the last W cycles | Smooths noise; reveals gradual drift |
| Rolling Std Dev | Variability over the last W cycles | Captures instability and erratic behaviour |
| EMA (Exponential Moving Average) | Weighted average, recent cycles weighted more | Reacts faster to sudden degradation |

`min_periods=1` was set for rolling mean and std so that early cycles (where
fewer than W readings exist) are not dropped. Rolling std NaN values at cycle 1
were filled with 0.

**Total rolling feature columns produced:**  
21 sensors × 3 window sizes × 3 statistics = **189 new columns**

All rolling operations used `groupby("unit_nr").transform(...)` to keep each
engine's history completely isolated.

---

## 4. Lag Features

Lag features record what a sensor read N cycles ago. They allow the model to
directly observe whether a sensor is rising, falling, or stable — information
that is invisible from the current reading alone.

Lags of **t−1** and **t−2** were added for all 21 sensor columns.  
NaN values produced for the first 1–2 cycles of each unit were filled with 0.

**Total lag feature columns produced:**  
21 sensors × 2 lag steps = **42 new columns**

---

## 5. Final Feature Matrix

After running the full pipeline (`build_features()`), the DataFrame grew from
**29 columns to 260 columns**:

| Component | Columns |
|-----------|---------|
| Original columns (sensors + settings + labels) | 29 |
| Rolling features added | +189 |
| Lag features added | +42 |
| **Total** | **260** |

The 4 non-feature columns (`unit_nr`, `time_cycles`, `max_cycles`, `RUL`) were
dropped before modelling, leaving **256 feature columns** for the model.

The final DataFrame was serialised to `data/features_df.pkl` using
`joblib.dump(..., compress=3)` for efficient storage and fast loading.

---

## 6. Train / Test Split

The dataset was split **by machine unit**, not by row shuffle. Shuffling rows in
time-series data would leak future sensor values into training, producing
artificially inflated evaluation scores. The last 20% of engine units were held
out for testing.

| Split | Units | Rows | Positive Rate (failure=1) |
|-------|-------|------|--------------------------|
| Train | 80 | 19,468 | ~14.2% |
| Test | 20 | 5,172 | ~15.1% |

The positive rate (~14–15%) confirms class imbalance, which was addressed during
modelling using `scale_pos_weight` in XGBoost.

---

## 7. Key Design Decisions

**groupby before every rolling/lag operation** — The most important correctness
rule in this pipeline. Applying a rolling window across all rows without grouping
by `unit_nr` would mix the final cycles of one engine with the early cycles of
the next engine in the DataFrame, producing completely invalid features. Every
single transform in this module uses `df.groupby("unit_nr")[col].transform(...)`.

**pd.concat instead of column-by-column assignment** — Features were accumulated
in a dictionary and concatenated once per function call using `pd.concat`. This
avoids the Pandas `PerformanceWarning` that occurs when hundreds of columns are
added to a DataFrame one at a time.

**Three window sizes (5, 10, 20)** — Short windows (5 cycles) capture rapid
sensor changes; medium windows (10 cycles) capture week-scale trends; long
windows (20 cycles) capture slow degradation patterns. Providing all three gives
the model the flexibility to learn which time horizon matters most for each
sensor.
