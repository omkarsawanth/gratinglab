"""
examples/analytical_demo.py
===========================
Comparison of N-slit interference patterns for N = 2, 5, 20.
"""

from __future__ import annotations
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib.pyplot as plt
import numpy as np
from gratinglab import analytical
from gratinglab import constants


def main() -> None:
    wavelength = constants.WAVELENGTH_RED_650
    pitch_d = constants.PITCH_CD
    slit_width_a = 0.4e-6
    slit_counts = [2, 5, 20]

    theta_deg = np.linspace(-45.0, 45.0, 3000)
    theta_rad = np.radians(theta_deg)

    envelope = np.sinc((slit_width_a * np.sin(theta_rad)) / wavelength) ** 2

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2b5c8f", "#d95f02", "#1b9e77"]

    for N, color in zip(slit_counts, colors):
        intensity = analytical.n_slit_intensity(
            theta=theta_rad,
            a=slit_width_a,
            d=pitch_d,
            N=N,
            wavelength=wavelength,
        )
        ax.plot(
            theta_deg,
            intensity,
            label=f"N = {N} slits",
            color=color,
            linewidth=1.8 if N == 20 else 1.3,
            alpha=0.9,
        )

    ax.plot(
        theta_deg,
        envelope,
        label=r"Single-slit envelope $\mathrm{sinc}^2(\beta/\pi)$",
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )

    orders, _ = analytical.order_angles(pitch_d, wavelength, m_max=2)
    for m, ang_rad in orders.items():
        if m != 0:
            ang_deg = np.degrees(ang_rad)
            ax.axvline(ang_deg, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
            ax.text(ang_deg, 1.02, f"m={m}", ha="center", va="bottom", fontsize=9, color="#333333")

    ax.set_title(
        f"GratingLab: Fraunhofer N-Slit Diffraction ($d = 1.6\\,\\mu\\mathrm{{m}}$, $\\lambda = 650\\,\\mathrm{{nm}}$, $a = 0.4\\,\\mu\\mathrm{{m}}$)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Diffraction Angle $\\theta$ [degrees]", fontsize=11)
    ax.set_ylabel("Normalized Intensity $I(\\theta) / I_0$", fontsize=11)
    ax.set_xlim(-45.0, 45.0)
    ax.set_ylim(0.0, 1.1)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    plt.tight_layout()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "analytical_demo.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Analytical demo figure saved to: {output_path}")


if __name__ == "__main__":
    main()
