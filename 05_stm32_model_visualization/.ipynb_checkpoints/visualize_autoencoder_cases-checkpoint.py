"""
Visualización funcional del autoencoder usado en STM32Cube.AI.

Objetivo:
- Usar el mismo modelo Keras entrenado que se importó en STM32Cube.AI.
- Seleccionar dos ventanas temporales reales:
    1) una con decisión NO_SEND
    2) una con decisión SEND
- Exportar datos para memoria/anexo:
    - ventanas temporales originales
    - vector x[14]
    - entrada normalizada
    - reconstrucción del autoencoder
    - error MSE
    - decisión final
    - figuras comparando entrada vs reconstrucción

Ejecutar desde la raíz del proyecto:
    python 05_stm32_model_visualization/visualize_autoencoder_cases.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_NAMES = [
    "mean_theta_x", "mean_theta_y", "mean_theta_z",
    "sigma_theta_x", "sigma_theta_y", "sigma_theta_z",
    "delta_theta_x", "delta_theta_y", "delta_theta_z",
    "range_theta_x", "range_theta_y", "range_theta_z",
    "rms_acc_activity", "mean_temperature",
]

FEATURE_LABELS = [
    "mean θx", "mean θy", "mean θz",
    "σ θx", "σ θy", "σ θz",
    "Δθx", "Δθy", "Δθz",
    "range θx", "range θy", "range θz",
    "RMS activity", "mean T",
]

ANGLE_COLS = ["X_reading_deg", "Y_reading_deg", "Z_reading_deg"]
TEMP_COL = "temperature_C"
STATUS_COL = "row_status"
SENSOR_COL = "sensor_id"
TIME_COL = "timestamp"
INCIDENT_COL = "incident_id"


def import_keras_model_loader():
    """
    Use tensorflow.keras first, because the .h5 model used for STM32Cube.AI
    was generated from the Keras/TensorFlow flow.
    """
    try:
        from tensorflow import keras  # type: ignore
        return keras
    except Exception as exc:
        raise RuntimeError(
            "Could not import tensorflow.keras. Install TensorFlow first."
        ) from exc


def window_status_to_label(status_values) -> int:
    """0 = normal / NO_SEND expected, 1 = anomalous / SEND expected."""
    values = [str(v).strip().upper() for v in status_values]
    return 0 if all(v == "OK" for v in values) else 1


def rms_activity_from_tilt(theta: np.ndarray) -> float:
    """
    Same approximation used in the training package when raw acceleration is not available.
    In real firmware, RMS|a| should come from raw acceleration samples.
    """
    if theta.shape[0] < 2:
        return 0.0
    diff = np.diff(theta, axis=0)
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def compute_x14_from_window(window: pd.DataFrame) -> np.ndarray:
    """Compute x[14] in the same order used during training."""
    theta = window[ANGLE_COLS].to_numpy(dtype=np.float32)
    temp = window[TEMP_COL].to_numpy(dtype=np.float32)

    mean_theta = theta.mean(axis=0)
    sigma_theta = theta.std(axis=0, ddof=0)
    delta_theta = theta[-1, :] - theta[0, :]
    range_theta = theta.max(axis=0) - theta.min(axis=0)
    rms_activity = rms_activity_from_tilt(theta)
    mean_temp = float(temp.mean())

    x = np.concatenate([
        mean_theta,
        sigma_theta,
        delta_theta,
        range_theta,
        np.array([rms_activity, mean_temp], dtype=np.float32),
    ]).astype(np.float32)

    if x.shape != (14,):
        raise RuntimeError(f"Expected 14 features, got {x.shape}")

    return x


def build_windows(csv_path: Path, window_size: int, stride: int):
    """
    Build temporal windows from the historical CSV.

    Returns:
        X_raw: array (n_windows, 14)
        y: labels 0/1 from row_status
        meta: dataframe with timestamps and status
        windows: list of raw window DataFrames
    """
    df = pd.read_csv(csv_path)

    required = set(ANGLE_COLS + [TEMP_COL, STATUS_COL, SENSOR_COL, TIME_COL, INCIDENT_COL])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values([INCIDENT_COL, SENSOR_COL, TIME_COL]).reset_index(drop=True)

    X_rows = []
    y_rows = []
    meta_rows = []
    windows = []

    for (incident_id, sensor_id), g in df.groupby([INCIDENT_COL, SENSOR_COL], sort=False):
        g = g.sort_values(TIME_COL).reset_index(drop=True)
        n = len(g)

        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            w = g.iloc[start:end].copy()

            x = compute_x14_from_window(w)
            y = window_status_to_label(w[STATUS_COL].tolist())

            X_rows.append(x)
            y_rows.append(y)
            windows.append(w)

            meta_rows.append({
                "window_index": len(X_rows) - 1,
                "incident_id": incident_id,
                "sensor_id": sensor_id,
                "start_timestamp": w[TIME_COL].iloc[0],
                "end_timestamp": w[TIME_COL].iloc[-1],
                "label_from_database": y,
                "status_values": "|".join(map(str, w[STATUS_COL].tolist())),
                "n_rows": len(w),
            })

    X_raw = np.vstack(X_rows).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.int64)
    meta = pd.DataFrame(meta_rows)

    return X_raw, y, meta, windows


def load_normalization_params(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    mean = np.asarray(data["mean"], dtype=np.float32)
    std = np.asarray(data["std"], dtype=np.float32)

    if mean.shape != (14,) or std.shape != (14,):
        raise ValueError("normalization_params.json must contain mean/std with 14 values")

    return mean, std


def load_threshold(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if "threshold_mse_normalized_space" in data:
        return float(data["threshold_mse_normalized_space"])
    if "threshold" in data:
        return float(data["threshold"])
    raise ValueError("Could not find threshold in ae_threshold.json")


def select_cases(meta: pd.DataFrame, y: np.ndarray, mse: np.ndarray, decision: np.ndarray):
    """
    Select one representative NO_SEND and one representative SEND case.

    Preference:
    - NO_SEND: database label normal and model decision NO_SEND, with low error.
    - SEND: database label anomalous and model decision SEND, with high error.
    """
    no_candidates = np.where((y == 0) & (decision == 0))[0]
    if len(no_candidates) == 0:
        no_candidates = np.where(decision == 0)[0]
    if len(no_candidates) == 0:
        raise RuntimeError("No NO_SEND case found. Check threshold/model.")

    send_candidates = np.where((y == 1) & (decision == 1))[0]
    if len(send_candidates) == 0:
        send_candidates = np.where(decision == 1)[0]
    if len(send_candidates) == 0:
        raise RuntimeError("No SEND case found. Check threshold/model.")

    # Pick a very clear normal case and a very clear anomalous/SEND case.
    no_idx = int(no_candidates[np.argmin(mse[no_candidates])])
    send_idx = int(send_candidates[np.argmax(mse[send_candidates])])

    return {
        "NO_SEND": no_idx,
        "SEND": send_idx,
    }


def save_case_outputs(
    case_name: str,
    idx: int,
    output_dir: Path,
    raw_window: pd.DataFrame,
    x_raw: np.ndarray,
    x_norm: np.ndarray,
    x_hat_norm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    mse_value: float,
    threshold: float,
    decision_value: int,
    meta_row: pd.Series,
):
    case_dir = output_dir / case_name.lower()
    case_dir.mkdir(parents=True, exist_ok=True)

    # Raw temporal window for annex.
    raw_window.to_csv(case_dir / f"{case_name.lower()}_temporal_window_raw.csv", index=False)

    # Feature and reconstruction table.
    x_hat_raw = (x_hat_norm * std + mean).astype(np.float32)
    per_feature_se = (x_norm - x_hat_norm) ** 2

    table = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "feature_label": FEATURE_LABELS,
        "x_raw": x_raw,
        "x_norm": x_norm,
        "x_reconstructed_norm": x_hat_norm,
        "x_reconstructed_raw_estimated": x_hat_raw,
        "squared_error_norm": per_feature_se,
    })
    table.to_csv(case_dir / f"{case_name.lower()}_x14_reconstruction.csv", index=False)

    # Small summary.
    summary = {
        "case": case_name,
        "window_index": int(idx),
        "incident_id": str(meta_row["incident_id"]),
        "sensor_id": str(meta_row["sensor_id"]),
        "start_timestamp": str(meta_row["start_timestamp"]),
        "end_timestamp": str(meta_row["end_timestamp"]),
        "label_from_database": int(meta_row["label_from_database"]),
        "status_values": str(meta_row["status_values"]),
        "mse_normalized_space": float(mse_value),
        "threshold": float(threshold),
        "decision": "SEND" if decision_value == 1 else "NO_SEND",
    }
    pd.DataFrame([summary]).to_csv(case_dir / f"{case_name.lower()}_summary.csv", index=False)
    (case_dir / f"{case_name.lower()}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    # Plot normalized input vs reconstructed output.
    x_axis = np.arange(len(FEATURE_NAMES))
    width = 0.38

    fig = plt.figure(figsize=(12, 5))
    plt.bar(x_axis - width / 2, x_norm, width, label="Entrada normalitzada x[14]")
    plt.bar(x_axis + width / 2, x_hat_norm, width, label="Reconstrucció autoencoder x_hat[14]")
    plt.axhline(0, linewidth=0.8)
    plt.xticks(x_axis, FEATURE_LABELS, rotation=45, ha="right")
    plt.ylabel("Valor normalitzat")
    plt.title(
        f"Cas {case_name}: MSE={mse_value:.6g}, "
        f"llindar={threshold:.6g}, decisió={'SEND' if decision_value == 1 else 'NO_SEND'}"
    )
    plt.legend()
    plt.tight_layout()
    fig.savefig(case_dir / f"{case_name.lower()}_reconstruction_plot.png", dpi=200)
    plt.close(fig)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/tiltmonitor_incidence_series_24h.csv")
    parser.add_argument("--model", default="artifacts/stm32cube_ai/autoencoder.h5")
    parser.add_argument("--norm", default="artifacts/common/normalization_params.json")
    parser.add_argument("--threshold", default="artifacts/common/ae_threshold.json")
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--out", default="artifacts/model_visualization")
    args = parser.parse_args()

    root = Path.cwd()
    csv_path = root / args.csv
    model_path = root / args.model
    norm_path = root / args.norm
    threshold_path = root / args.threshold
    output_dir = root / args.out
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Autoencoder functional visualization")
    print("=" * 70)
    print(f"CSV:        {csv_path}")
    print(f"Model:      {model_path}")
    print(f"Norm:       {norm_path}")
    print(f"Threshold:  {threshold_path}")
    print(f"Output dir: {output_dir}")
    print()

    for p in [csv_path, model_path, norm_path, threshold_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    print("Loading model...")
    keras = import_keras_model_loader()
    model = keras.models.load_model(model_path, compile=False)
    model.summary()

    print("\nLoading normalization and threshold...")
    mean, std = load_normalization_params(norm_path)
    threshold = load_threshold(threshold_path)
    print(f"Threshold MSE: {threshold:.9g}")

    print("\nBuilding windows and x[14] features...")
    X_raw, y, meta, windows = build_windows(csv_path, args.window_size, args.stride)
    print(f"Windows: {len(X_raw)}")

    X_norm = ((X_raw - mean) / std).astype(np.float32)

    print("\nRunning autoencoder inference in Python...")
    X_hat_norm = model.predict(X_norm, batch_size=256, verbose=0).astype(np.float32)
    mse = np.mean((X_norm - X_hat_norm) ** 2, axis=1).astype(np.float32)
    decision = (mse > threshold).astype(np.int64)

    cases = select_cases(meta, y, mse, decision)

    summaries = []
    for case_name, idx in cases.items():
        summaries.append(
            save_case_outputs(
                case_name=case_name,
                idx=idx,
                output_dir=output_dir,
                raw_window=windows[idx],
                x_raw=X_raw[idx],
                x_norm=X_norm[idx],
                x_hat_norm=X_hat_norm[idx],
                mean=mean,
                std=std,
                mse_value=float(mse[idx]),
                threshold=threshold,
                decision_value=int(decision[idx]),
                meta_row=meta.iloc[idx],
            )
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "selected_cases_summary.csv", index=False)

    print("\nSelected cases")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    print("\nGenerated files:")
    print(f"  {output_dir / 'selected_cases_summary.csv'}")
    print(f"  {output_dir / 'no_send' / 'no_send_temporal_window_raw.csv'}")
    print(f"  {output_dir / 'no_send' / 'no_send_x14_reconstruction.csv'}")
    print(f"  {output_dir / 'no_send' / 'no_send_reconstruction_plot.png'}")
    print(f"  {output_dir / 'send' / 'send_temporal_window_raw.csv'}")
    print(f"  {output_dir / 'send' / 'send_x14_reconstruction.csv'}")
    print(f"  {output_dir / 'send' / 'send_reconstruction_plot.png'}")


if __name__ == "__main__":
    main()
