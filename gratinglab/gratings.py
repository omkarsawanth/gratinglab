"""
gratinglab.gratings
===================
Complex optical transmittance functions on 1D and 2D spatial coordinate grids.

Transmittance is represented as a complex scalar field:
    t(x) = A(x) * exp(i * phi(x))
"""

from __future__ import annotations
import math
import warnings
from typing import Literal
import numpy as np


class Grid1D:
    """1D spatial coordinate grid helper for diffraction simulations."""

    def __init__(self, width: float, num_samples: int) -> None:
        if width <= 0:
            raise ValueError("Grid width must be positive.")
        if num_samples < 2:
            raise ValueError("num_samples must be >= 2.")

        self.width: float = float(width)
        self.num_samples: int = int(num_samples)
        self.dx: float = self.width / self.num_samples
        self.x: np.ndarray = (
            np.arange(self.num_samples, dtype=float) - (self.num_samples / 2.0)
        ) * self.dx

    def check_resolution(self, d: float) -> None:
        if d <= 0:
            raise ValueError("Grating period d must be positive.")

        min_resolution_dx = d / 20.0
        if self.dx > min_resolution_dx:
            warnings.warn(
                f"Grating period under-resolved: dx ({self.dx:.3e} m) > d/20 ({min_resolution_dx:.3e} m). "
                f"At least 20 samples per grating period are recommended.",
                UserWarning,
                stacklevel=2,
            )

        min_window_width = 10.0 * d
        if self.width < min_window_width:
            warnings.warn(
                f"Computational window too narrow: width ({self.width:.3e} m) < 10*d ({min_window_width:.3e} m). "
                f"Window holds fewer than 10 grating periods.",
                UserWarning,
                stacklevel=2,
            )


class Grid2D:
    """2D spatial coordinate grid helper."""

    def __init__(
        self,
        width_x: float,
        num_samples_x: int,
        width_y: float | None = None,
        num_samples_y: int | None = None,
    ) -> None:
        w_y = width_x if width_y is None else width_y
        n_y = num_samples_x if num_samples_y is None else num_samples_y

        self.grid_x: Grid1D = Grid1D(width_x, num_samples_x)
        self.grid_y: Grid1D = Grid1D(w_y, n_y)
        self.dx: float = self.grid_x.dx
        self.dy: float = self.grid_y.dx
        self.X, self.Y = np.meshgrid(self.grid_x.x, self.grid_y.x)

    def check_resolution(self, dx_period: float, dy_period: float) -> None:
        self.grid_x.check_resolution(dx_period)
        self.grid_y.check_resolution(dy_period)


def binary_amplitude(
    x: np.ndarray,
    d: float,
    a: float,
) -> np.ndarray:
    if d <= 0 or a <= 0:
        raise ValueError("d and a must be positive.")
    if a > d:
        raise ValueError("Slit width 'a' cannot exceed pitch 'd'.")

    x_rel = np.mod(x + (d / 2.0), d) - (d / 2.0)
    transmittance = np.zeros(x.shape, dtype=np.complex128)
    in_slit = np.abs(x_rel) <= (a / 2.0)
    transmittance[in_slit] = 1.0 + 0.0j
    return transmittance


def sinusoidal_phase(
    x: np.ndarray,
    d: float,
    phi0: float,
) -> np.ndarray:
    if d <= 0:
        raise ValueError("Grating pitch 'd' must be positive.")
    phase = phi0 * np.sin((2.0 * math.pi * x) / d)
    return np.exp(1j * phase).astype(np.complex128)


def blazed(
    x: np.ndarray,
    d: float,
    phase_depth: float,
) -> np.ndarray:
    if d <= 0:
        raise ValueError("Grating pitch 'd' must be positive.")
    x_mod = np.mod(x, d)
    phase = phase_depth * (x_mod / d)
    return np.exp(1j * phase).astype(np.complex128)


def binary_phase(
    x: np.ndarray,
    d: float,
    duty: float,
    phi: float,
) -> np.ndarray:
    if d <= 0:
        raise ValueError("Grating pitch 'd' must be positive.")
    if not (0.0 < duty < 1.0):
        raise ValueError("Duty cycle must be strictly in the range (0.0, 1.0).")

    x_rel = np.mod(x, d)
    in_ridge = x_rel < (duty * d)
    transmittance = np.ones(x.shape, dtype=np.complex128)
    transmittance[in_ridge] = np.exp(1j * phi)
    return transmittance


def grating_2d(
    x: np.ndarray,
    y: np.ndarray,
    dx_period: float,
    dy_period: float,
    kind: Literal["rect_aperture", "sinusoidal"] = "rect_aperture",
    aperture_x: float | None = None,
    aperture_y: float | None = None,
    phi0: float = math.pi,
) -> np.ndarray:
    if dx_period <= 0 or dy_period <= 0:
        raise ValueError("Lattice periods must be positive.")

    if x.ndim == 1 and y.ndim == 1:
        X, Y = np.meshgrid(x, y)
    elif x.ndim == 2 and y.ndim == 2:
        X, Y = x, y
    else:
        raise ValueError("x and y must be both 1D arrays or both 2D meshgrids.")

    if kind == "rect_aperture":
        ax = 0.5 * dx_period if aperture_x is None else aperture_x
        ay = 0.5 * dy_period if aperture_y is None else aperture_y
        tx = binary_amplitude(X, dx_period, ax)
        ty = binary_amplitude(Y, dy_period, ay)
        return (tx * ty).astype(np.complex128)

    elif kind == "sinusoidal":
        phase = (phi0 / 2.0) * (
            np.sin((2.0 * math.pi * X) / dx_period)
            + np.sin((2.0 * math.pi * Y) / dy_period)
        )
        return np.exp(1j * phase).astype(np.complex128)

    else:
        raise ValueError(f"Unknown 2D grating kind: '{kind}'. Choose 'rect_aperture' or 'sinusoidal'.")


def finite_aperture(
    x: np.ndarray,
    n_lines: int,
    d: float,
) -> np.ndarray:
    if n_lines < 1:
        raise ValueError("n_lines must be >= 1.")
    if d <= 0:
        raise ValueError("Grating pitch 'd' must be positive.")

    w_aperture = n_lines * d
    mask = np.zeros(x.shape, dtype=float)
    mask[np.abs(x) <= (w_aperture / 2.0)] = 1.0
    return mask
