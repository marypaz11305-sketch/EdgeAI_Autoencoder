"""
Genera tablas visuales para memoria/TFG a partir de los resultados ya creados por:
05_stm32_model_visualization/visualize_autoencoder_cases.py

Entrada esperada:
- artifacts/model_visualization/no_send/no_send_x14_reconstruction.csv
- artifacts/model_visualization/send/send_x14_reconstruction.csv

Salida:
- tablas CSV resumidas
- tablas PNG visuales para insertar en la memoria
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ROOT = Path.cwd()
BASE_RESULTS = ROOT / "artifacts" / "model_visualization"

FEATURE_COLS = [
    "feature_label",
    "x_norm",
    "x_reconstructed_norm",
    "squared_error_norm",
]

RENAME_MAP = {
    "feature_label": "Característica",
    "x_norm": "Input normalitzat",
    "x_reconstructed_norm": "Reconstrucció",
    "squared_error_norm": "Error quadràtic",
}


def load_case(case_name: str) -> pd.DataFrame:
    path = BASE_RESULTS / case_name / f"{case_name}_x14_reconstruction.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")
    df = pd.read_csv(path)
    df = df[FEATURE_COLS].copy()
    df = df.rename(columns=RENAME_MAP)
    df["Diferència abs."] = (df["Input normalitzat"] - df["Reconstrucció"]).abs()
    return df


def round_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Input normalitzat", "Reconstrucció", "Diferència abs.", "Error quadràtic"]:
        out[col] = out[col].map(lambda x: f"{x:.4f}")
    return out


def save_table_png(df_numeric: pd.DataFrame, case_name: str):
    df_display = round_table(df_numeric)

    fig_height = max(5.5, 0.42 * len(df_display) + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    # Header formatting
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(cell.get_height() * 1.15)

    # Highlight the 4 largest absolute differences
    diff = df_numeric["Diferència abs."].to_numpy()
    top_idx = set(np.argsort(diff)[-4:].tolist())

    diff_col = list(df_display.columns).index("Diferència abs.")
    err_col = list(df_display.columns).index("Error quadràtic")

    for i in range(len(df_display)):
        row = i + 1  # row 0 is header
        if i in top_idx:
            table[(row, diff_col)].set_text_props(weight="bold")
            table[(row, err_col)].set_text_props(weight="bold")

    title = (
        f"Cas {case_name.upper()}: comparació entre input i reconstrucció "
        f"(espai normalitzat)"
    )
    ax.set_title(title, fontsize=11, pad=16)

    out_path = BASE_RESULTS / case_name / f"{case_name}_comparison_table.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return out_path


def main():
    print("Generating visual comparison tables...")
    print(f"Results dir: {BASE_RESULTS}")

    for case_name in ["no_send", "send"]:
        df = load_case(case_name)

        csv_path = BASE_RESULTS / case_name / f"{case_name}_comparison_table.csv"
        df.to_csv(csv_path, index=False)

        png_path = save_table_png(df, case_name)

        print(f"[OK] {case_name}:")
        print(f"     CSV -> {csv_path}")
        print(f"     PNG -> {png_path}")

    print("Done.")


if __name__ == "__main__":
    main()
