"""
validation.cd_dvd_validation
============================
Validation harness comparing optical bench laboratory measurements of commercial
Compact Discs (CD) and Digital Versatile Discs (DVD) against GratingLab simulations.

Governing Physics (Reflection Grating, Normal Incidence):
1. Geometric diffraction angle:
   theta_meas = arctan(spot_offset / L)

2. Grating pitch calculation:
   d_meas = (m * lambda) / sin(theta_meas)

3. Simulated prediction:
   theta_pred = arcsin(m * lambda / d_standard)
   offset_pred = L * tan(theta_pred)

Safety Warning:
NEVER look directly into the primary or specularly reflected laser beam.
Use diffuse projection screens and Class 2 / 3R compliant laser safety goggles.
"""

from __future__ import annotations
import argparse
import csv
import math
import os
import sys

# Ensure project root is in sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
from gratinglab import analytical
from gratinglab import constants


STANDARD_PITCHES: dict[str, float] = {
    "CD": constants.PITCH_CD,      # 1.60 um
    "DVD": constants.PITCH_DVD,    # 0.74 um
}


def load_measurements(csv_path: str) -> list[dict[str, float | str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Measurement dataset not found: {csv_path}")

    records: list[dict[str, float | str]] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            if header is None:
                header = [h.strip() for h in row]
                continue

            row_dict: dict[str, float | str] = {}
            for col_name, val_str in zip(header, row):
                val_clean = val_str.strip()
                if col_name == "media":
                    row_dict["media"] = val_clean.upper()
                else:
                    row_dict[col_name] = float(val_clean)

            if "wavelength_nm" not in row_dict:
                row_dict["wavelength_nm"] = 650.0

            records.append(row_dict)

    if not records:
        raise ValueError(f"No valid measurement records parsed from: {csv_path}")
    return records


def process_measurements(records: list[dict[str, float | str]]) -> list[dict]:
    results: list[dict] = []

    for r in records:
        media = str(r["media"])
        L = float(r["screen_distance_m"])
        m = int(r["order_m"])
        offset_meas = float(r["spot_offset_m"])
        wl = float(r["wavelength_nm"]) * 1e-9

        d_std = STANDARD_PITCHES.get(media, constants.PITCH_CD)

        theta_meas = math.atan(offset_meas / L)
        sin_theta_meas = math.sin(theta_meas)

        d_meas = (m * wl) / sin_theta_meas if sin_theta_meas > 0 else float("nan")

        predicted_orders, _ = analytical.order_angles(d=d_std, wavelength=wl, m_max=abs(m))
        theta_pred = predicted_orders.get(m, float("nan"))
        offset_pred = L * math.tan(theta_pred) if not math.isnan(theta_pred) else float("nan")

        err_angle_pct = abs(theta_meas - theta_pred) / theta_pred * 100.0 if not math.isnan(theta_pred) else float("nan")
        err_pitch_pct = abs(d_meas - d_std) / d_std * 100.0

        results.append({
            "media": media,
            "L_m": L,
            "order": m,
            "offset_meas_m": offset_meas,
            "offset_pred_m": offset_pred,
            "theta_meas_deg": math.degrees(theta_meas),
            "theta_pred_deg": math.degrees(theta_pred) if not math.isnan(theta_pred) else float("nan"),
            "d_meas_um": d_meas * 1e6,
            "d_std_um": d_std * 1e6,
            "err_angle_pct": err_angle_pct,
            "err_pitch_pct": err_pitch_pct,
            "wavelength_nm": r["wavelength_nm"],
        })

    return results


def print_validation_report(results: list[dict]) -> None:
    print("\n" + "=" * 98)
    print("  GRATINGLAB: CD / DVD OPTICAL BENCH VALIDATION REPORT")
    print("=" * 98)
    print(f"{'Media':<5} | {'m':>2} | {'L (m)':>5} | {'y_meas (cm)':>11} | {'y_pred (cm)':>11} | {'θ_meas (°)':>10} | {'θ_pred (°)':>10} | {'d_meas (µm)':>11} | {'d_std':>7} | {'Err (%)':>7}")
    print("-" * 98)

    by_media: dict[str, list[float]] = {}

    for row in results:
        media = row["media"]
        by_media.setdefault(media, []).append(row["d_meas_um"])

        print(
            f"{media:<5} | "
            f"{row['order']:2d} | "
            f"{row['L_m']:5.3f} | "
            f"{row['offset_meas_m']*100:10.2f}  | "
            f"{row['offset_pred_m']*100:10.2f}  | "
            f"{row['theta_meas_deg']:9.2f}° | "
            f"{row['theta_pred_deg']:9.2f}° | "
            f"{row['d_meas_um']:10.3f}  | "
            f"{row['d_std_um']:6.2f}  | "
            f"{row['err_pitch_pct']:6.2f}%"
        )

    print("-" * 98)
    print("Summary Pitch Statistics (Mean ± 1σ):")
    for media, pitches in by_media.items():
        std_val = STANDARD_PITCHES[media] * 1e6
        mean_p = float(np.mean(pitches))
        sigma_p = float(np.std(pitches, ddof=1)) if len(pitches) > 1 else 0.0
        bias_pct = abs(mean_p - std_val) / std_val * 100.0
        print(f"  • {media:3s}:  Measured d = {mean_p:.4f} ± {sigma_p:.4f} µm  |  Standard = {std_val:.2f} µm  |  Mean Error = {bias_pct:.2f}%")
    print("=" * 98 + "\n")


def plot_validation_results(results: list[dict], output_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    cd_pts = [r for r in results if r["media"] == "CD"]
    dvd_pts = [r for r in results if r["media"] == "DVD"]

    y_pred_cd = [r["offset_pred_m"] * 100 for r in cd_pts]
    y_meas_cd = [r["offset_meas_m"] * 100 for r in cd_pts]
    y_pred_dvd = [r["offset_pred_m"] * 100 for r in dvd_pts]
    y_meas_dvd = [r["offset_meas_m"] * 100 for r in dvd_pts]

    max_val = max(max(y_pred_cd + y_meas_cd, default=1.0), max(y_pred_dvd + y_meas_dvd, default=1.0)) * 1.08

    ax1.plot([0, max_val], [0, max_val], "k--", label="Ideal Agreement (1:1)", lw=1.3)
    if cd_pts:
        ax1.scatter(y_pred_cd, y_meas_cd, color="#1f77b4", s=60, label="CD Points ($d=1.6\\,\\mu\\mathrm{m}$)", zorder=4)
    if dvd_pts:
        ax1.scatter(y_pred_dvd, y_meas_dvd, color="#d62728", marker="s", s=60, label="DVD Points ($d=0.74\\,\\mu\\mathrm{m}$)", zorder=4)

    ax1.set_title("Diffraction Spot Offset: Predicted vs Measured", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Predicted Spot Position $y_{\\mathrm{pred}}$ [cm]", fontsize=10)
    ax1.set_ylabel("Measured Spot Position $y_{\\mathrm{meas}}$ [cm]", fontsize=10)
    ax1.set_xlim(0, max_val)
    ax1.set_ylim(0, max_val)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    indices = np.arange(len(results))
    labels = [f"{r['media']} m={r['order']}\n(L={r['L_m']:.2f}m)" for r in results]
    meas_pitches = [r["d_meas_um"] for r in results]
    std_pitches = [r["d_std_um"] for r in results]

    width = 0.35
    ax2.bar(indices - width / 2, std_pitches, width, label="Standard Industry Pitch", color="#7f7f7f", alpha=0.7)
    ax2.bar(indices + width / 2, meas_pitches, width, label="Measured Pitch $d_{\\mathrm{meas}}$", color="#2ca02c", alpha=0.85)

    ax2.set_xticks(indices)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Track Pitch $d$ [µm]", fontsize=10)
    ax2.set_title("Experimental Track Pitch Extraction", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Validation plot exported to: {output_path}")


def run_validation(csv_path: str, output_img: str | None = None) -> list[dict]:
    if output_img is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_img = os.path.join(project_root, "examples", "outputs", "cd_dvd_validation.png")

    records = load_measurements(csv_path)
    results = process_measurements(records)
    print_validation_report(results)
    plot_validation_results(results, output_img)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="GratingLab: CD/DVD Optical Bench Experimental Validation")
    default_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_cd_dvd_data.csv")
    parser.add_argument("--csv", type=str, default=default_csv, help="Path to experimental CSV measurement file.")
    parser.add_argument("--output", type=str, default=None, help="Path to output comparison plot.")
    args = parser.parse_args()

    run_validation(args.csv, args.output)


if __name__ == "__main__":
    main()
