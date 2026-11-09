"""
gratinglab.case_studies
=======================
Advanced material characterization case studies:
1. study_strain_temperature
2. study_groove_depth
3. study_2d_lattice
"""

from __future__ import annotations
import math
import os
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from gratinglab import characterize
from gratinglab import constants
from gratinglab import gratings
from gratinglab import propagation


def _ensure_output_dir() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    out_dir = os.path.join(project_root, "examples", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def study_strain_temperature(
    d0: float = constants.PITCH_CD,
    wavelength: float = constants.WAVELENGTH_RED_650,
    alpha_thermal: float = 17.0e-6,
    num_steps: int = 11,
    noise_level: float = 0.005,
    save_fig: bool = True,
) -> dict[str, np.ndarray]:
    print("=" * 82)
    print("  Case Study 1: Optical Strain & Temperature Characterization")
    print("=" * 82)

    applied_strains = np.linspace(0.0, 0.01, num_steps)
    recovered_strains = np.zeros(num_steps)
    recovered_pitches = np.zeros(num_steps)
    temp_changes = applied_strains / alpha_thermal

    grid_width = 60.0 * d0
    grid = gratings.Grid1D(width=grid_width, num_samples=4096)
    expected_orders = np.array([-2, -1, 0, 1, 2])

    for i, eps in enumerate(applied_strains):
        d_strained = d0 * (1.0 + eps)
        # Use 35% duty cycle so orders 1 and 2 are both well-represented (not missing)
        a_strained = 0.35 * d_strained

        t_field = gratings.binary_amplitude(grid.x, d=d_strained, a=a_strained)

        sin_theta, intensity, _ = propagation.far_field_fft(
            t_field,
            dx=grid.dx,
            wavelength=wavelength,
            pad_factor=2,
        )

        if noise_level > 0:
            noise = np.random.normal(0.0, noise_level, size=intensity.shape)
            intensity = np.clip(intensity + noise, 0.0, None)

        peak_sin, _, _ = characterize.find_orders(sin_theta, intensity, prominence_rel=0.04, min_distance=8)

        matched_sin = []
        for m in expected_orders:
            target = m * (wavelength / d_strained)
            idx = np.argmin(np.abs(peak_sin - target))
            matched_sin.append(peak_sin[idx])

        d_est, _ = characterize.estimate_period(np.array(matched_sin), expected_orders, wavelength)
        recovered_pitches[i] = d_est
        recovered_strains[i] = (d_est - d0) / d0

    print(f"{'Applied ε (%)':>14} | {'ΔT (°C)':>10} | {'True d (µm)':>12} | {'Recov d (µm)':>12} | {'Recov ε (%)':>13} | {'Rel Error':>11}")
    print("-" * 84)
    for i in range(num_steps):
        print(
            f"{applied_strains[i]*100:13.3f}% | "
            f"{temp_changes[i]:10.1f} | "
            f"{d0*(1.0+applied_strains[i])*1e6:12.4f} | "
            f"{recovered_pitches[i]*1e6:12.4f} | "
            f"{recovered_strains[i]*100:12.3f}% | "
            f"{abs(recovered_strains[i]-applied_strains[i])*100:10.4f}%"
        )
    print("=" * 84 + "\n")

    if save_fig:
        out_dir = _ensure_output_dir()
        fig_path = os.path.join(out_dir, "case_study_1_strain_temperature.png")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(applied_strains * 100, applied_strains * 100, "k--", label="Ideal 1:1 Response", lw=1.5)
        ax1.plot(applied_strains * 100, recovered_strains * 100, "ro-", label="Recovered Strain", markersize=5, lw=1.2)
        ax1.set_xlabel("Applied Uniaxial Strain $\\epsilon$ [%]", fontsize=11)
        ax1.set_ylabel("Diffraction-Recovered Strain $\\epsilon_{\\mathrm{rec}}$ [%]", fontsize=11)
        ax1.set_title("Optical Strain Tracking ($d_0 = 1.6\\,\\mu\\mathrm{m}$)", fontsize=11, fontweight="bold")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left")

        pitch_residual_nm = (recovered_pitches - d0 * (1.0 + applied_strains)) * 1e9
        ax2.bar(applied_strains * 100, pitch_residual_nm, width=0.06, color="#1f77b4", alpha=0.85, edgecolor="black")
        ax2.axhline(0.0, color="gray", linestyle="--", lw=1.0)
        ax2.set_xlabel("Applied Strain $\\epsilon$ [%]", fontsize=11)
        ax2.set_ylabel("Pitch Residual $\\Delta d$ [nm]", fontsize=11)
        ax2.set_title("Sub-Nanometer Pitch Measurement Error", fontsize=11, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"[OK] Case study 1 figure saved to: {fig_path}")

    return {
        "applied_strains": applied_strains,
        "recovered_strains": recovered_strains,
        "recovered_pitches": recovered_pitches,
    }


def study_groove_depth(
    wavelength: float = constants.WAVELENGTH_RED_650,
    n_index: float = 1.50,
    unknown_depth_nm: float = 520.0,
    save_fig: bool = True,
) -> dict[str, float]:
    print("=" * 82)
    print("  Case Study 2: Sinusoidal Grating Groove Depth Characterization")
    print("=" * 82)

    h_sweep_nm = np.linspace(0.0, 1200.0, 300)
    h_sweep_m = h_sweep_nm * 1e-9
    phi0_sweep = (math.pi * (n_index - 1.0) * h_sweep_m) / wavelength

    eta_0 = special.jv(0, phi0_sweep) ** 2
    eta_1 = special.jv(1, phi0_sweep) ** 2
    eta_2 = special.jv(2, phi0_sweep) ** 2

    h_true = unknown_depth_nm * 1e-9
    phi0_true = (math.pi * (n_index - 1.0) * h_true) / wavelength

    orders = np.array([-2, -1, 0, 1, 2])
    true_efficiencies = special.jv(orders, phi0_true) ** 2

    np.random.seed(101)
    measured_efficiencies = true_efficiencies + np.random.normal(0.0, 0.015, size=orders.shape)
    measured_efficiencies = np.clip(measured_efficiencies, 0.0, None)

    h_rec, phi0_rec, h_err = characterize.estimate_groove_depth(
        efficiencies=measured_efficiencies,
        orders=orders,
        wavelength=wavelength,
        n_index=n_index,
    )

    error_pct = abs(h_rec - h_true) / h_true * 100.0

    print(f"Substrate Index n:       {n_index:.2f}")
    print(f"Laser Wavelength:        {wavelength*1e9:.1f} nm")
    print("-" * 82)
    print(f"{'Parameter':<22} | {'True Value':>15} | {'Recovered Value':>18} | {'Error (%)':>10}")
    print("-" * 82)
    print(f"{'Phase Modulation (φ₀)':<22} | {phi0_true:14.4f} rad | {phi0_rec:15.4f} rad | {abs(phi0_rec-phi0_true)/phi0_true*100:9.2f}%")
    print(f"{'Groove Depth (h)':<22} | {h_true*1e9:14.2f} nm  | {h_rec*1e9:15.2f} nm  | {error_pct:9.2f}%")
    print(f"{'Fit Uncertainty (1σ)':<22} | {'---':>15} | {h_err*1e9:15.2f} nm  | {'---':>10}")
    print("=" * 82 + "\n")

    if save_fig:
        out_dir = _ensure_output_dir()
        fig_path = os.path.join(out_dir, "case_study_2_groove_depth.png")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(h_sweep_nm, eta_0, label=r"Order 0: $\eta_0 = J_0(\phi_0)^2$", color="#1f77b4", lw=2.0)
        ax.plot(h_sweep_nm, eta_1, label=r"Order $\pm 1$: $\eta_1 = J_1(\phi_0)^2$", color="#ff7f0e", lw=2.0)
        ax.plot(h_sweep_nm, eta_2, label=r"Order $\pm 2$: $\eta_2 = J_2(\phi_0)^2$", color="#2ca02c", lw=2.0)

        ax.axvline(unknown_depth_nm, color="crimson", ls="--", lw=1.5, label=f"True Depth $h = {unknown_depth_nm:.1f}\\,\\mathrm{{nm}}$")
        ax.scatter([h_rec * 1e9], [special.jv(1, phi0_rec)**2], color="darkred", s=80, zorder=5, label=f"Recovered $h = {h_rec*1e9:.1f}\\,\\mathrm{{nm}}$")

        ax.set_title("Diffraction Efficiency vs Groove Depth for Sinusoidal Phase Grating", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Physical Groove Depth $h$ [nm]", fontsize=11)
        ax.set_ylabel("Diffraction Efficiency $\\eta_m$", fontsize=11)
        ax.set_xlim(0.0, 1200.0)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", frameon=True, fontsize=10)

        plt.tight_layout()
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"[OK] Case study 2 figure saved to: {fig_path}")

    return {
        "true_depth_nm": unknown_depth_nm,
        "recovered_depth_nm": h_rec * 1e9,
        "error_pct": error_pct,
    }


def study_2d_lattice(
    dx_true: float = 2.5e-6,
    dy_true: float = 1.8e-6,
    wavelength: float = constants.WAVELENGTH_RED_650,
    save_fig: bool = True,
) -> dict[str, float]:
    print("=" * 82)
    print("  Case Study 3: 2D Photonic Lattice Constant Characterization")
    print("=" * 82)

    grid_size = 25.0e-6
    n_pts = 256
    grid_2d = gratings.Grid2D(width_x=grid_size, num_samples_x=n_pts)

    ax_ap = 0.5 * dx_true
    ay_ap = 0.5 * dy_true

    t_2d = gratings.grating_2d(
        x=grid_2d.grid_x.x,
        y=grid_2d.grid_y.x,
        dx_period=dx_true,
        dy_period=dy_true,
        kind="rect_aperture",
        aperture_x=ax_ap,
        aperture_y=ay_ap,
    )

    sin_x, sin_y, intensity_2d = propagation.far_field_fft_2d(
        field_2d=t_2d,
        dx=grid_2d.dx,
        dy=grid_2d.dy,
        wavelength=wavelength,
    )

    center_y = len(sin_y) // 2
    center_x = len(sin_x) // 2

    intensity_x_slice = intensity_2d[center_y, :]
    intensity_y_slice = intensity_2d[:, center_x]

    peaks_x_sin, _, _ = characterize.find_orders(sin_x, intensity_x_slice, prominence_rel=0.02, min_distance=3)
    peaks_y_sin, _, _ = characterize.find_orders(sin_y, intensity_y_slice, prominence_rel=0.02, min_distance=3)

    pos_x = peaks_x_sin[peaks_x_sin > 0.05]
    pos_y = peaks_y_sin[peaks_y_sin > 0.05]

    sin_x1 = float(np.min(pos_x))
    sin_y1 = float(np.min(pos_y))

    peaks_2d = [
        (0.0, 0.0),
        (sin_x1, 0.0),
        (-sin_x1, 0.0),
        (0.0, sin_y1),
        (0.0, -sin_y1),
    ]

    dx_rec, dy_rec = characterize.lattice_constants_2d(peaks_2d, wavelength=wavelength)

    err_x_pct = abs(dx_rec - dx_true) / dx_true * 100.0
    err_y_pct = abs(dy_rec - dy_true) / dy_true * 100.0

    print(f"Laser Wavelength:        {wavelength*1e9:.1f} nm")
    print("-" * 82)
    print(f"{'Axis':<10} | {'True Pitch (µm)':>16} | {'Recovered Pitch (µm)':>22} | {'Error (%)':>12}")
    print("-" * 82)
    print(f"{'X-Axis (d_x)':<10} | {dx_true*1e6:15.4f}  | {dx_rec*1e6:21.4f}  | {err_x_pct:11.3f}%")
    print(f"{'Y-Axis (d_y)':<10} | {dy_true*1e6:15.4f}  | {dy_rec*1e6:21.4f}  | {err_y_pct:11.3f}%")
    print("=" * 82 + "\n")

    if save_fig:
        out_dir = _ensure_output_dir()
        fig_path = os.path.join(out_dir, "case_study_3_2d_lattice.png")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

        extent_real = [grid_2d.grid_x.x[0] * 1e6, grid_2d.grid_x.x[-1] * 1e6, grid_2d.grid_y.x[0] * 1e6, grid_2d.grid_y.x[-1] * 1e6]
        ax1.imshow(np.abs(t_2d), extent=extent_real, cmap="binary_r", origin="lower")
        ax1.set_title(f"Real-Space 2D Grating Mask ($d_x = {dx_true*1e6:.2f}\\,\\mu\\mathrm{{m}}$, $d_y = {dy_true*1e6:.2f}\\,\\mu\\mathrm{{m}}$)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Transverse $x$ [µm]", fontsize=10)
        ax1.set_ylabel("Transverse $y$ [µm]", fontsize=10)
        ax1.set_xlim(-6.0, 6.0)
        ax1.set_ylim(-6.0, 6.0)

        extent_diff = [sin_x[0], sin_x[-1], sin_y[0], sin_y[-1]]
        im2 = ax2.imshow(
            np.log10(np.maximum(intensity_2d, 1e-4)),
            extent=extent_diff,
            cmap="inferno",
            origin="lower",
        )
        ax2.scatter([sin_x1, -sin_x1, 0, 0], [0, 0, sin_y1, -sin_y1], color="cyan", s=40, facecolors="none", lw=1.5, label="Detected Orders")
        ax2.set_title("2D Far-Field Diffraction Pattern (Log Scale)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Direction Sine $\\sin\\theta_x$", fontsize=10)
        ax2.set_ylabel("Direction Sine $\\sin\\theta_y$", fontsize=10)
        ax2.legend(loc="upper right", fontsize=9)
        fig.colorbar(im2, ax=ax2, label="$\\log_{10}(I / I_0)$")

        plt.tight_layout()
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        print(f"[OK] Case study 3 figure saved to: {fig_path}")

    return {
        "dx_true": dx_true,
        "dx_rec": dx_rec,
        "dy_true": dy_true,
        "dy_rec": dy_rec,
        "error_x_pct": err_x_pct,
        "error_y_pct": err_y_pct,
    }


def run_all_case_studies() -> None:
    print("\n" + "#" * 84)
    print("  GRATINGLAB: ADVANCED MATERIAL CHARACTERIZATION CASE STUDIES")
    print("#" * 84 + "\n")

    study_strain_temperature()
    study_groove_depth()
    study_2d_lattice()

    out_dir = _ensure_output_dir()
    print(f"All case study plots successfully generated in:\n  {out_dir}\n")


if __name__ == "__main__":
    run_all_case_studies()
