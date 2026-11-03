#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Feature engineering for the G7 Tiltmeter autoencoder case study.

The generated input vector has 14 features, in the same order used in the TFG:

x = [
    mean(theta_x), mean(theta_y), mean(theta_z),
    sigma(theta_x), sigma(theta_y), sigma(theta_z),
    delta(theta_x), delta(theta_y), delta(theta_z),
    range(theta_x), range(theta_y), range(theta_z),
    RMS|a|, mean(T)
]

Notes:
- The provided 24 h CSV contains processed tilt readings, not raw ax/ay/az acceleration.
- Therefore RMS|a| is approximated as a window activity descriptor from tilt variations.
- In a real ADXL355 acquisition campaign, RMS|a| should be computed from raw acceleration samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "mean_theta_x", "mean_theta_y", "mean_theta_z",
    "sigma_theta_x", "sigma_theta_y", "sigma_theta_z",
    "delta_theta_x", "delta_theta_y", "delta_theta_z",
    "range_theta_x", "range_theta_y", "range_theta_z",
    "rms_acc_activity", "mean_temperature",
]

ANGLE_COLS = ["X_reading_deg", "Y_reading_deg", "Z_reading_deg"]
TEMP_COL = "temperature_C"
STATUS_COL = "row_status"
SENSOR_COL = "sensor_id"
TIME_COL = "timestamp"


@dataclass(frozen=True)
class WindowConfig:
    window_size: int = 4
    stride: int = 1


def _window_status_to_label(status_values: Iterable[str]) -> int:
    """0 = normal / NO_SEND, 1 = anomalous / SEND."""
    values = [str(v).strip().upper() for v in status_values]
    return 0 if all(v == "OK" for v in values) else 1


def _rms_activity_from_tilt(theta: np.ndarray) -> float:
    """
    Approximation of RMS|a| when raw acceleration is not available.

    For real firmware/data, replace this with RMS(sqrt(ax^2 + ay^2 + az^2)).
    Here we use the RMS of inter-sample tilt differences across the three axes.
    """
    if theta.shape[0] < 2:
        return 0.0
    diff = np.diff(theta, axis=0)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def compute_x14_from_window(window: pd.DataFrame) -> np.ndarray:
    """Compute the 14-feature vector from a single temporal window."""
    theta = window[ANGLE_COLS].to_numpy(dtype=np.float32)
    temp = window[TEMP_COL].to_numpy(dtype=np.float32)

    mean_theta = theta.mean(axis=0)
    sigma_theta = theta.std(axis=0, ddof=0)
    delta_theta = theta[-1, :] - theta[0, :]
    range_theta = theta.max(axis=0) - theta.min(axis=0)
    rms_activity = _rms_activity_from_tilt(theta)
    mean_temp = float(temp.mean())

    x = np.concatenate([
        mean_theta,
        sigma_theta,
        delta_theta,
        range_theta,
        np.array([rms_activity, mean_temp], dtype=np.float32),
    ]).astype(np.float32)

    if x.shape != (14,):
        raise RuntimeError(f"Internal feature error: expected 14 features, got {x.shape}")

    return x


def build_windows_from_csv(csv_path: str | Path, cfg: WindowConfig) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Read the CSV and build X14/y windows.

    Returns:
        X: float32 array, shape (n_windows, 14)
        y: int array, shape (n_windows,), 0=OK/normal, 1=incidence/anomaly
        meta: DataFrame with sensor_id, timestamps and labels per window
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required = set(ANGLE_COLS + [TEMP_COL, STATUS_COL, SENSOR_COL, TIME_COL])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(["incident_id", SENSOR_COL, TIME_COL]).reset_index(drop=True)

    X_rows = []
    y_rows = []
    meta_rows = []

    group_cols = ["incident_id", SENSOR_COL]
    for (incident_id, sensor_id), g in df.groupby(group_cols, sort=False):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        n = len(g)
        for start in range(0, n - cfg.window_size + 1, cfg.stride):
            end = start + cfg.window_size
            w = g.iloc[start:end]
            x = compute_x14_from_window(w)
            y = _window_status_to_label(w[STATUS_COL].tolist())

            X_rows.append(x)
            y_rows.append(y)
            meta_rows.append({
                "incident_id": incident_id,
                "sensor_id": sensor_id,
                "start_timestamp": w[TIME_COL].iloc[0],
                "end_timestamp": w[TIME_COL].iloc[-1],
                "label": y,
                "status_values": "|".join(map(str, w[STATUS_COL].tolist())),
            })

    X = np.vstack(X_rows).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.int64)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


def save_feature_csv(X: np.ndarray, y: np.ndarray, meta: pd.DataFrame, output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    feature_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    feature_df.insert(0, "label", y)
    out = pd.concat([meta.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    out.to_csv(output_csv, index=False)


# In[ ]:




