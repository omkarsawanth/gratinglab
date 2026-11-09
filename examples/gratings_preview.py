"""
examples/gratings_preview.py
============================
Visual comparison of the 4 fundamental 1D optical grating profiles.
"""

from __future__ import annotations
import math
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import matplotlib.pyplot as plt
import numpy as np
from gratinglab import gratings


def main() -> None:
    d = 1.6e-6
    width = 6.0 * d
    num_samples = 2400

    grid = gratings.Grid1D(width=width, num_samples=num_samples)
    x_um = grid.x * 1e6
    d_um = d * 1e6

    t_bin_amp = gratings.binary_amplitude(grid.x, d=d, a=0.8e-6)
    t_sin_phase = gratings.sinusoidal_phase(grid.x, d=d, phi0=math.pi)
    t_blazed = gratings.blazed(grid.x, d=d, phase_depth=2.0 * math.pi)
    t_bin_phase = gratings.binary_phase(grid.x, d=d, duty=0.5, phi=math.pi)

    profiles = [
        ("Binary Amplitude Grating (a = 0.8 µm)", t_bin_amp, "#1f77b4"),
        ("Sinusoidal Phase Grating (φ₀ = π rad)", t_sin_phase, "#ff7f0e"),
        ("Blazed Sawtooth Phase Grating (Δφ = 2π rad)", t_blazed, "#2ca02c"),
        ("Binary Phase Grating (duty = 50%, Δφ = π rad)", t_bin_phase, "#d62728"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)

    for i, (title, t, color) in enumerate(profiles):
        amp = np.abs(t)
        phase = np.angle(t)

        ax_amp = axes[i, 0]
        ax_amp.plot(x_um, amp, color=color, linewidth=1.8)
        ax_amp.set_ylabel("Amplitude $|t(x)|$", fontsize=10)
        ax_amp.set_ylim(-0.1, 1.2)
        ax_amp.grid(True, linestyle="--", alpha=0.5)
        ax_amp.set_title(f"{title} - Amplitude", fontsize=11, fontweight="bold")

        for p in range(-3, 4):
            ax_amp.axvline(p * d_um, color="gray", linestyle=":", alpha=0.4)

        ax_phase = axes[i, 1]
        ax_phase.plot(x_um, phase, color=color, linewidth=1.8)
        ax_phase.set_ylabel("Phase [rad]", fontsize=10)
        ax_phase.set_ylim(-1.2 * math.pi, 2.2 * math.pi)
        ax_phase.set_yticks([-math.pi, 0, math.pi, 2 * math.pi])
        ax_phase.set_yticklabels([r"$-\pi$", "0", r"$\pi$", r"$2\pi$"])
        ax_phase.grid(True, linestyle="--", alpha=0.5)
        ax_phase.set_title(f"{title} - Phase", fontsize=11, fontweight="bold")

        for p in range(-3, 4):
            ax_phase.axvline(p * d_um, color="gray", linestyle=":", alpha=0.4)

    axes[3, 0].set_xlabel("Spatial Coordinate $x$ [µm]", fontsize=11)
    axes[3, 1].set_xlabel("Spatial Coordinate $x$ [µm]", fontsize=11)

    plt.tight_layout()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "gratings_preview.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Gratings preview figure saved to: {output_path}")


if __name__ == "__main__":
    main()
