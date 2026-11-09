"""
examples/talbot_demo.py
=======================
Talbot Carpet Simulation (Near-Field Wave Propagation).
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
from gratinglab import gratings
from gratinglab import propagation


def main() -> None:
    wavelength = constants.WAVELENGTH_RED_650
    pitch_d = constants.PITCH_CD
    slit_width_a = 0.8e-6

    z_t = analytical.talbot_distance(d=pitch_d, wavelength=wavelength)

    total_width = 30.0 * pitch_d
    num_samples = 3000
    grid = gratings.Grid1D(width=total_width, num_samples=num_samples)

    t_grating = gratings.binary_amplitude(grid.x, d=pitch_d, a=slit_width_a)

    x, z_array, carpet = propagation.talbot_carpet(
        field=t_grating,
        dx=grid.dx,
        wavelength=wavelength,
        d=pitch_d,
        n_z_steps=250,
        z_max=2.0 * z_t,
    )

    x_crop_mask = np.abs(x) <= (3.0 * pitch_d)
    x_crop_um = x[x_crop_mask] * 1e6
    carpet_crop = carpet[:, x_crop_mask]
    z_um = z_array * 1e6
    z_t_um = z_t * 1e6

    fig, ax = plt.subplots(figsize=(10, 7))

    im = ax.imshow(
        carpet_crop,
        extent=[x_crop_um[0], x_crop_um[-1], z_um[0], z_um[-1]],
        origin="lower",
        aspect="auto",
        cmap="inferno",
    )

    ax.axhline(z_t_um, color="cyan", linestyle="--", linewidth=1.5, label=f"Talbot Plane $z_T = {z_t_um:.2f}\\,\\mu\\mathrm{{m}}$")
    ax.axhline(2.0 * z_t_um, color="cyan", linestyle=":", linewidth=1.5, label=f"$2 z_T = {2.0*z_t_um:.2f}\\,\\mu\\mathrm{{m}}$")
    ax.axhline(0.5 * z_t_um, color="lime", linestyle="--", linewidth=1.2, label=f"Half-Talbot $z_T/2$ (shifted $d/2$)")
    ax.axhline(0.25 * z_t_um, color="yellow", linestyle=":", linewidth=1.2, label=f"Quarter-Talbot $z_T/4$ (freq. doubling)")

    ax.set_title(
        f"GratingLab: Near-Field Talbot Carpet ($d = 1.6\\,\\mu\\mathrm{{m}}$, $\\lambda = 650\\,\\mathrm{{nm}}$)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Transverse Coordinate $x$ [µm]", fontsize=11)
    ax.set_ylabel("Propagation Distance $z$ [µm]", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.85, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative Intensity $|U(x, z)|^2$", fontsize=10)

    plt.tight_layout()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "talbot_demo.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Talbot demo figure saved to: {output_path}")


if __name__ == "__main__":
    main()
