"""
gratinglab.characterize
=======================
Advanced material characterization module:
Extracts microscopic grating parameters from simulated or experimental
diffraction patterns using peak detection, linear regression, and non-linear
least-squares curve fitting.
"""

from __future__ import annotations
import math
import numpy as np
from scipy import optimize, signal, special, stats


def find_orders(
    sin_theta: np.ndarray,
    intensity: np.ndarray,
    prominence_rel: float = 0.05,
    min_distance: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(sin_theta) != len(intensity):
        raise ValueError("sin_theta and intensity must have identical lengths.")

    max_intensity = float(np.max(intensity))
    if max_intensity <= 0:
        return np.array([]), np.array([]), np.array([], dtype=int)

    abs_prominence = prominence_rel * max_intensity
    indices, _ = signal.find_peaks(
        intensity,
        prominence=abs_prominence,
        distance=min_distance,
    )

    peak_sin = sin_theta[indices]
    peak_heights = intensity[indices]

    return peak_sin, peak_heights, indices


def estimate_period(
    peak_positions_sin: np.ndarray,
    orders: np.ndarray,
    wavelength: float,
    theta_i: float = 0.0,
) -> tuple[float, float]:
    if len(peak_positions_sin) < 2:
        raise ValueError("At least 2 diffraction peaks are required for linear regression.")
    if wavelength <= 0:
        raise ValueError("Wavelength must be positive.")

    y = np.asarray(peak_positions_sin, dtype=float) - math.sin(theta_i)
    x = np.asarray(orders, dtype=float)

    res = stats.linregress(x, y)
    slope = float(res.slope)
    slope_std_err = float(res.stderr) if res.stderr is not None else 0.0

    if abs(slope) < 1e-15:
        raise ValueError("Regression slope is near zero; cannot determine grating pitch.")

    d_estimated = wavelength / slope
    d_uncertainty = abs(d_estimated * (slope_std_err / slope))

    return abs(d_estimated), d_uncertainty


def estimate_duty_cycle(
    peak_heights: np.ndarray,
    orders: np.ndarray,
) -> tuple[float, float]:
    if len(peak_heights) < 3:
        raise ValueError("At least 3 orders are required to fit duty cycle.")

    m_arr = np.asarray(orders, dtype=float)
    y_arr = np.asarray(peak_heights, dtype=float)

    def sinc2_model(m: np.ndarray, duty: float, i0: float) -> np.ndarray:
        return i0 * (np.sinc(m * duty) ** 2)

    i0_init = float(np.max(y_arr))
    p0 = [0.5, i0_init]
    bounds = ([0.02, 0.0], [0.98, 3.0 * i0_init])

    popt, pcov = optimize.curve_fit(
        sinc2_model,
        m_arr,
        y_arr,
        p0=p0,
        bounds=bounds,
        maxfev=10000,
    )

    fitted_duty = float(popt[0])
    duty_uncertainty = float(np.sqrt(pcov[0, 0])) if pcov is not None else 0.0

    return fitted_duty, duty_uncertainty


def estimate_groove_depth(
    efficiencies: np.ndarray,
    orders: np.ndarray,
    wavelength: float,
    n_index: float,
) -> tuple[float, float, float]:
    if n_index <= 1.0:
        raise ValueError("Refractive index n_index must be > 1.0.")
    if wavelength <= 0:
        raise ValueError("Wavelength must be positive.")
    if len(efficiencies) < 2:
        raise ValueError("At least 2 diffraction orders are required.")

    m_arr = np.asarray(orders, dtype=int)
    y_arr = np.asarray(efficiencies, dtype=float)

    def bessel_model(m: np.ndarray, phi0: float, scale: float) -> np.ndarray:
        return scale * (special.jv(m, phi0) ** 2)

    total_eff = float(np.sum(y_arr))
    p0 = [1.0, total_eff if total_eff > 0 else 1.0]
    bounds = ([0.01, 0.0], [10.0, 5.0])

    popt, pcov = optimize.curve_fit(
        bessel_model,
        m_arr,
        y_arr,
        p0=p0,
        bounds=bounds,
        maxfev=10000,
    )

    fitted_phi0 = float(popt[0])
    phi0_uncertainty = float(np.sqrt(pcov[0, 0])) if pcov is not None else 0.0

    conversion_factor = wavelength / (math.pi * (n_index - 1.0))
    groove_depth = fitted_phi0 * conversion_factor
    depth_uncertainty = phi0_uncertainty * conversion_factor

    return groove_depth, fitted_phi0, depth_uncertainty


def lattice_constants_2d(
    peaks_2d_sin: list[tuple[float, float]] | np.ndarray,
    wavelength: float,
) -> tuple[float, float]:
    pts = np.asarray(peaks_2d_sin, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("peaks_2d_sin must have shape (N, 2).")

    x_axis_peaks = pts[(np.abs(pts[:, 1]) < 0.05) & (pts[:, 0] > 0.05)]
    if len(x_axis_peaks) == 0:
        raise ValueError("Could not find order (1, 0) along X axis.")
    sin_x1 = float(np.min(x_axis_peaks[:, 0]))
    dx = wavelength / sin_x1

    y_axis_peaks = pts[(np.abs(pts[:, 0]) < 0.05) & (pts[:, 1] > 0.05)]
    if len(y_axis_peaks) == 0:
        raise ValueError("Could not find order (0, 1) along Y axis.")
    sin_y1 = float(np.min(y_axis_peaks[:, 1]))
    dy = wavelength / sin_y1

    return dx, dy
