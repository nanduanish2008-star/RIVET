"""Sensor CSV validation, unusual-reading flags, and clearly labelled simulated data."""
import numpy as np
import pandas as pd

FIELDS = {"turbidity_ntu": (0, 4000), "water_level_cm": (0, 1000),
          "ph": (0, 14), "temperature_c": (-5, 50)}


def load_sensor_csv(source):
    try:
        df = pd.read_csv(source)
    except Exception as exc:
        raise ValueError(f"Could not read the CSV: {exc}") from exc
    if "timestamp" not in df.columns:
        raise ValueError("The CSV needs a 'timestamp' column.")
    present = [f for f in FIELDS if f in df.columns]
    if not present:
        raise ValueError("No recognised sensor columns. Expected any of: " + ", ".join(FIELDS))
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    report = {"rows_read": int(len(df)), "bad_timestamps": int(df["timestamp"].isna().sum()),
              "missing": {}, "out_of_range": {}}
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows have a valid timestamp.")
    for f in present:
        df[f] = pd.to_numeric(df[f], errors="coerce")
        lo, hi = FIELDS[f]
        report["missing"][f] = int(df[f].isna().sum())
        report["out_of_range"][f] = int(((df[f] < lo) | (df[f] > hi)).sum())
    df["t_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    return df, present, report


def flag_unusual(df, field, window=15, z=3.5):
    """Robust rolling z-score: flag readings far from the local median, in local MAD units."""
    s = df[field]
    med = s.rolling(window, min_periods=3, center=True).median()
    dev = (s - med).abs()
    scale = (dev.rolling(window, min_periods=3, center=True).median() * 1.4826).clip(lower=1e-6)
    return (dev / scale > z).astype(bool)


def make_simulated_csv(hours=3, step_min=5, seed=7):
    """SIMULATED DATA - NOT LIVE SENSOR MEASUREMENTS. Includes one injected turbidity event."""
    rng = np.random.default_rng(seed)
    n = int(hours * 60 / step_min)
    ts = pd.date_range("2026-01-01 10:00", periods=n, freq=f"{step_min}min")
    turb = 20 + rng.normal(0, 1.5, n)
    turb[n // 2: n // 2 + 3] += 80
    df = pd.DataFrame({
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "turbidity_ntu": turb.round(2),
        "water_level_cm": (50 + rng.normal(0, 0.4, n)).round(2),
        "ph": (7.2 + rng.normal(0, 0.05, n)).round(2),
        "temperature_c": (24 + rng.normal(0, 0.3, n)).round(2),
    })
    return df.to_csv(index=False)
