"""
main.py
=======
CLI entry point for GratingLab:
Parses command-line arguments and launches the interactive Matplotlib simulation studio.

Usage:
    python main.py --grating binary --wavelength 650 --period 1.6 --slits 10 --screen-dist 0.5
    python main.py --grating blazed --wavelength 532 --period 3.33
"""

from __future__ import annotations
import argparse
import os
import sys

# Ensure an interactive backend is used for desktop GUI rendering
import matplotlib
if matplotlib.get_backend().lower() in ("agg", "template", "pdf", "ps", "svg"):
    for _backend in ["TkAgg", "QtAgg", "Qt5Agg", "WXAgg"]:
        try:
            matplotlib.use(_backend, force=True)
            break
        except Exception:
            continue

from gratinglab import visualize


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GratingLab: Laser Diffraction & Material Characterization Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--grating",
        choices=["binary", "sinusoidal", "blazed", "2d"],
        default="binary",
        help="Grating type profile.",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=650.0,
        help="Laser source wavelength in nanometers (400 - 700 nm).",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=1.6,
        help="Grating pitch / track spacing d in micrometers.",
    )
    parser.add_argument(
        "--slits",
        "-N",
        type=int,
        default=10,
        help="Number of illuminated grating slits/rulings.",
    )
    parser.add_argument(
        "--screen-dist",
        type=float,
        default=0.5,
        help="Distance from grating to observation screen in meters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    grating_mapping = {
        "binary": "Binary Amplitude",
        "sinusoidal": "Sinusoidal Phase",
        "blazed": "Blazed Phase",
        "2d": "2D Orthogonal",
    }
    selected_grating = grating_mapping[args.grating]

    print("=" * 70)
    print("  Launching GratingLab Interactive Studio")
    print("=" * 70)
    print(f"  Grating Profile:   {selected_grating}")
    print(f"  Laser Wavelength:  {args.wavelength:.1f} nm")
    print(f"  Grating Pitch d:   {args.period:.2f} µm")
    print(f"  Illuminated Slits: {args.slits}")
    print(f"  Screen Distance:   {args.screen_dist:.2f} m")
    print("=" * 70)

    visualize.launch_interactive_app(
        grating_type=selected_grating,
        wavelength_nm=args.wavelength,
        period_um=args.period,
        n_slits=args.slits,
        screen_dist_m=args.screen_dist,
    )


if __name__ == "__main__":
    main()
