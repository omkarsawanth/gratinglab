"""
scripts/generate_docs.py
========================
Automated benchmark and test report generator for GratingLab.
Executes 100 repetitions of each module function and writes:
- docs/benchmark.txt
- docs/test_report.txt
"""

import os
import subprocess
import sys
import timeit
import numpy as np


def run_benchmarks() -> str:
    setup_code = """
import sys
sys.path.insert(0, '.')
import numpy as np
from gratinglab import constants, analytical, gratings, propagation, characterize

wavelength = 650.0e-9
d = 1.6e-6
a = 0.8e-6
grid = gratings.Grid1D(width=60.0 * d, num_samples=2048)
t_field = gratings.binary_amplitude(grid.x, d=d, a=a)
sin_theta, intensity, theta_deg = propagation.far_field_fft(t_field, dx=grid.dx, wavelength=wavelength, pad_factor=2)
peaks_sin, heights, _ = characterize.find_orders(sin_theta, intensity)
orders = np.array([-2, -1, 0, 1, 2])
matched_sin = np.array([m * (wavelength / d) for m in orders])
"""

    benchmarks = [
        ("gratinglab.constants.nm_to_m", "constants.nm_to_m(650.0)", "Scalar float"),
        ("gratinglab.analytical.order_angles", "analytical.order_angles(d, wavelength, m_max=4)", "d=1.6µm, m_max=4"),
        ("gratinglab.analytical.n_slit_intensity", "analytical.n_slit_intensity(np.linspace(-0.5, 0.5, 2048), a, d, N=20, wavelength=wavelength)", "N=20, 2048 pts"),
        ("gratinglab.analytical.missing_orders", "analytical.missing_orders(d, a, m_max=20)", "d=2µm, a=1µm"),
        ("gratinglab.analytical.talbot_distance", "analytical.talbot_distance(d, wavelength)", "d=1.6µm, λ=650nm"),
        ("gratinglab.gratings.binary_amplitude", "gratings.binary_amplitude(grid.x, d=d, a=a)", "N=2048 samples"),
        ("gratinglab.gratings.sinusoidal_phase", "gratings.sinusoidal_phase(grid.x, d=d, phi0=3.14)", "N=2048 samples"),
        ("gratinglab.gratings.blazed", "gratings.blazed(grid.x, d=d, phase_depth=6.28)", "N=2048 samples"),
        ("gratinglab.propagation.gaussian_beam", "propagation.gaussian_beam(grid.x, w0=40e-6)", "N=2048 samples"),
        ("gratinglab.propagation.far_field_fft", "propagation.far_field_fft(t_field, dx=grid.dx, wavelength=wavelength, pad_factor=2)", "N=2048, pad=2"),
        ("gratinglab.propagation.angular_spectrum", "propagation.angular_spectrum(t_field, dx=grid.dx, wavelength=wavelength, z=10e-6)", "N=2048, 1 step"),
        ("gratinglab.characterize.find_orders", "characterize.find_orders(sin_theta, intensity)", "2048 samples"),
        ("gratinglab.characterize.estimate_period", "characterize.estimate_period(matched_sin, orders, wavelength)", "5 orders regression"),
    ]

    lines = [
        "=" * 88,
        "  GratingLab Module Runtime Benchmark (100 repetitions per function)",
        "=" * 88,
        f"{'Module & Function':<42} | {'Input Size':<18} | {'Mean Latency':>16}",
        "-" * 88,
    ]

    for name, stmt, desc in benchmarks:
        times = timeit.repeat(stmt=stmt, setup=setup_code, repeat=5, number=20)
        mean_time = np.mean(times) / 20.0
        if mean_time < 1e-3:
            time_str = f"{mean_time * 1e6:8.2f} µs"
        else:
            time_str = f"{mean_time * 1e3:8.2f} ms"
        lines.append(f"{name:<42} | {desc:<18} | {time_str:>16}")

    lines.append("=" * 88)
    return "\n".join(lines)


def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(project_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    print("Running 100-repetition timing benchmarks...")
    bench_output = run_benchmarks()
    bench_file = os.path.join(docs_dir, "benchmark.txt")
    with open(bench_file, "w", encoding="utf-8") as f:
        f.write(bench_output)
    print(f"[OK] Benchmark written to: {bench_file}")

    print("Running pytest with coverage...")
    test_file = os.path.join(docs_dir, "test_report.txt")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--cov=gratinglab", "--cov-report=term"],
        capture_output=True,
        text=True,
        cwd=project_root,
    )
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n" + proc.stderr)
    print(f"[OK] Test report written to: {test_file}")


if __name__ == "__main__":
    main()
