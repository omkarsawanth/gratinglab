"""
gratinglab.propagation
=======================
Numerical wave optics propagation engines:
1. Gaussian incident beam synthesis (1D and 2D).
2. Far-field Fraunhofer diffraction via Fast Fourier Transform (FFT).
3. Near-to-intermediate field wave propagation via the Angular Spectrum Method (ASM).
4. Talbot carpet generation (z-stack self-imaging).
5. 2D Fraunhofer diffraction via 2D FFT.
"""

from __future__ import annotations
import math
from typing import Literal
import numpy as np
from scipy import fft


def gaussian_beam(
    x: np.ndarray,
    w0: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    if w0 <= 0:
        raise ValueError("Beam waist radius w0 must be positive.")
    field = amplitude * np.exp(-(x**2) / (w0**2))
    return field.astype(np.complex128)


def gaussian_beam_2d(
    x: np.ndarray,
    y: np.ndarray,
    w0_x: float,
    w0_y: float | None = None,
    amplitude: float = 1.0,
) -> np.ndarray:
    if w0_x <= 0:
        raise ValueError("w0_x must be positive.")
    w_y = w0_x if w0_y is None else w0_y
    if w_y <= 0:
        raise ValueError("w0_y must be positive.")

    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y)
    else:
        X, Y = x, y

    field_2d = amplitude * np.exp(-((X**2) / (w0_x**2) + (Y**2) / (w_y**2)))
    return field_2d.astype(np.complex128)


def far_field_fft(
    field: np.ndarray,
    dx: float,
    wavelength: float,
    pad_factor: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dx <= 0 or wavelength <= 0:
        raise ValueError("dx and wavelength must be positive.")
    if pad_factor < 1:
        raise ValueError("pad_factor must be >= 1.")

    num_samples = len(field)
    padded_len = num_samples * pad_factor

    if pad_factor > 1:
        pad_total = padded_len - num_samples
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        u_in = np.pad(field, (pad_left, pad_right), mode="constant")
    else:
        u_in = field

    field_shifted = fft.ifftshift(u_in)
    u_fft = fft.fftshift(fft.fft(field_shifted)) * dx

    fx = fft.fftshift(fft.fftfreq(padded_len, d=dx))
    sin_theta_raw = wavelength * fx
    prop_mask = np.abs(sin_theta_raw) <= 1.0

    sin_theta = sin_theta_raw[prop_mask]
    u_prop = u_fft[prop_mask]
    intensity_raw = np.abs(u_prop) ** 2

    max_val = np.max(intensity_raw) if len(intensity_raw) > 0 else 0.0
    intensity = intensity_raw / max_val if max_val > 0 else intensity_raw

    theta_rad = np.arcsin(sin_theta)
    theta_deg = np.degrees(theta_rad)

    return sin_theta, intensity, theta_deg


def sin_theta_to_screen_position(
    theta_rad: np.ndarray,
    screen_distance_l: float,
) -> np.ndarray:
    if screen_distance_l <= 0:
        raise ValueError("Screen distance L must be positive.")
    return screen_distance_l * np.tan(theta_rad)


def angular_spectrum(
    field: np.ndarray,
    dx: float,
    wavelength: float,
    z: float,
    theta_i: float = 0.0,
    evanescent: Literal["zero", "decay"] = "zero",
) -> np.ndarray:
    if dx <= 0 or wavelength <= 0:
        raise ValueError("dx and wavelength must be positive.")

    num_samples = len(field)
    x = (np.arange(num_samples, dtype=float) - (num_samples / 2.0)) * dx

    k = (2.0 * math.pi) / wavelength
    if abs(theta_i) > 1e-12:
        phase_tilt = np.exp(1j * k * math.sin(theta_i) * x)
        u_initial = field * phase_tilt
    else:
        u_initial = field

    if abs(z) < 1e-15:
        return u_initial.copy()

    fx = fft.fftshift(fft.fftfreq(num_samples, d=dx))
    inv_lambda_sq = (1.0 / wavelength) ** 2
    fx_sq = fx**2

    propagating_mask = fx_sq <= inv_lambda_sq
    kz = np.zeros(num_samples, dtype=complex)
    kz[propagating_mask] = 2.0 * math.pi * np.sqrt(inv_lambda_sq - fx_sq[propagating_mask])

    h_prop = np.zeros(num_samples, dtype=complex)
    h_prop[propagating_mask] = np.exp(1j * kz[propagating_mask] * z)

    if evanescent == "decay":
        evan_mask = ~propagating_mask
        gamma = 2.0 * math.pi * np.sqrt(fx_sq[evan_mask] - inv_lambda_sq)
        h_prop[evan_mask] = np.exp(-gamma * abs(z))

    u_shifted = fft.ifftshift(u_initial)
    a_f = fft.fftshift(fft.fft(u_shifted))
    a_prop = a_f * h_prop
    u_out_shifted = fft.ifft(fft.ifftshift(a_prop))
    u_out = fft.fftshift(u_out_shifted)

    return u_out.astype(np.complex128)


def talbot_carpet(
    field: np.ndarray,
    dx: float,
    wavelength: float,
    d: float,
    n_z_steps: int = 200,
    z_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if d <= 0:
        raise ValueError("Grating pitch 'd' must be positive.")

    z_talbot = 2.0 * (d**2) / wavelength
    z_limit = 2.0 * z_talbot if z_max is None else z_max

    z_array = np.linspace(0.0, z_limit, n_z_steps)
    num_samples = len(field)
    x = (np.arange(num_samples, dtype=float) - (num_samples / 2.0)) * dx

    carpet = np.zeros((n_z_steps, num_samples), dtype=float)

    for i, z_val in enumerate(z_array):
        u_z = angular_spectrum(field, dx=dx, wavelength=wavelength, z=z_val)
        carpet[i, :] = np.abs(u_z) ** 2

    return x, z_array, carpet


def far_field_fft_2d(
    field_2d: np.ndarray,
    dx: float,
    dy: float,
    wavelength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dx <= 0 or dy <= 0 or wavelength <= 0:
        raise ValueError("dx, dy, and wavelength must be positive.")

    ny, nx = field_2d.shape

    u_shifted = fft.ifftshift(field_2d)
    u_fft2 = fft.fftshift(fft.fft2(u_shifted)) * (dx * dy)

    fx = fft.fftshift(fft.fftfreq(nx, d=dx))
    fy = fft.fftshift(fft.fftfreq(ny, d=dy))

    sin_theta_x_raw = wavelength * fx
    sin_theta_y_raw = wavelength * fy

    mask_x = np.abs(sin_theta_x_raw) <= 1.0
    mask_y = np.abs(sin_theta_y_raw) <= 1.0

    sin_theta_x = sin_theta_x_raw[mask_x]
    sin_theta_y = sin_theta_y_raw[mask_y]

    u_cropped = u_fft2[np.ix_(mask_y, mask_x)]
    intensity_2d_raw = np.abs(u_cropped) ** 2

    max_val = np.max(intensity_2d_raw) if intensity_2d_raw.size > 0 else 0.0
    intensity_2d = intensity_2d_raw / max_val if max_val > 0 else intensity_2d_raw

    return sin_theta_x, sin_theta_y, intensity_2d
