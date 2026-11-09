"""
gratinglab.analytical
=====================
Closed-form analytical models for optical diffraction gratings.

Key equations:
1. Generalized Grating Equation:
   d * (sin(theta_m) - sin(theta_i)) = m * lambda
   => sin(theta_m) = sin(theta_i) + (m * lambda / d)

2. Fraunhofer N-Slit Normalized Intensity Distribution:
   I(theta) = [sinc(beta / pi)]^2 * [sin(N * delta) / (N * sin(delta))]^2
   where:
     beta  = (pi * a / lambda) * (sin(theta) - sin(theta_i))   (diffraction envelope)
     delta = (pi * d / lambda) * (sin(theta) - sin(theta_i))   (N-slit interference)
     a     = individual slit aperture width [m]
     d     = grating pitch / slit spacing [m]
     N     = number of illuminated slits
     Limit as delta -> m * pi: [sin(N * delta) / (N * sin(delta))]^2 -> 1.0

3. Missing Orders Condition:
   m = p * (d / a) for integer p != 0

4. Chromatic Resolving Power (Rayleigh Criterion):
   R = lambda / Delta_lambda = |m| * N

5. Angular Dispersion:
   D = d(theta_m) / d(lambda) = m / (d * cos(theta_m))  [rad/m]

6. Talbot Self-Imaging Distance:
   z_T = 2 * d^2 / lambda  [m]
"""

from __future__ import annotations
import math
import numpy as np


def order_angles(
    d: float,
    wavelength: float,
    theta_i: float = 0.0,
    m_max: int | None = None,
) -> tuple[dict[int, float], list[int]]:
    """
    Compute diffraction angles theta_m for propagating orders and classify evanescent orders.

    Equation:
        d * (sin(theta_m) - sin(theta_i)) = m * lambda
        => sin(theta_m) = sin(theta_i) + m * (lambda / d)

    Propagating condition:
        |sin(theta_m)| <= 1.0

    Evanescent condition:
        |sin(theta_m)| > 1.0

    Parameters
    ----------
    d : float
        Grating pitch / track spacing in meters.
    wavelength : float
        Laser wavelength in vacuum in meters.
    theta_i : float, optional
        Angle of incidence in radians (default is 0.0, normal incidence).
    m_max : int | None, optional
        Maximum absolute order to evaluate. If None, evaluates from 0 up to
        the cutoff order including the first boundary evanescent orders.

    Returns
    -------
    tuple[dict[int, float], list[int]]
        - propagating_orders: Mapping of order m -> theta_m in radians.
        - evanescent_orders: List of integer orders m where |sin(theta_m)| > 1.0.

    Raises
    ------
    ValueError
        If d <= 0 or wavelength <= 0.
    """
    if d <= 0:
        raise ValueError("Grating pitch 'd' must be positive.")
    if wavelength <= 0:
        raise ValueError("Wavelength must be positive.")

    sin_theta_i = math.sin(theta_i)

    # Determine order evaluation range
    if m_max is None:
        pos_cutoff = int(math.floor((1.0 - sin_theta_i) * d / wavelength))
        neg_cutoff = int(math.floor((1.0 + sin_theta_i) * d / wavelength))
        eval_max = max(abs(pos_cutoff), abs(neg_cutoff)) + 1
    else:
        if m_max < 0:
            raise ValueError("m_max must be non-negative.")
        eval_max = m_max

    propagating_orders: dict[int, float] = {}
    evanescent_orders: list[int] = []

    for m in range(-eval_max, eval_max + 1):
        sin_theta_m = sin_theta_i + (m * wavelength / d)
        if abs(sin_theta_m) <= 1.0:
            propagating_orders[m] = math.asin(sin_theta_m)
        else:
            evanescent_orders.append(m)

    return propagating_orders, evanescent_orders


def n_slit_intensity(
    theta: float | np.ndarray,
    a: float,
    d: float,
    N: int,
    wavelength: float,
    theta_i: float = 0.0,
) -> float | np.ndarray:
    r"""
    Calculate normalized Fraunhofer diffraction intensity profile for an N-slit grating.

    Equation:
        I(theta) = I_envelope(beta) * I_interference(delta)
        I(theta) = [sinc(beta / pi)]^2 * [sin(N * delta) / (N * sin(delta))]^2

    where:
        beta  = (pi * a / lambda) * (sin(theta) - sin(theta_i))
        delta = (pi * d / lambda) * (sin(theta) - sin(theta_i))

    Limit behavior:
        As delta -> m * pi:
        lim [sin(N * delta) / (N * sin(delta))]^2 = 1.0

    Parameters
    ----------
    theta : float | np.ndarray
        Observation angle(s) in radians.
    a : float
        Slit width in meters.
    d : float
        Grating pitch in meters.
    N : int
        Number of illuminated slits (N >= 1).
    wavelength : float
        Wavelength in meters.
    theta_i : float, optional
        Incidence angle in radians (default 0.0).

    Returns
    -------
    float | np.ndarray
        Normalized intensity in [0, 1]. Returns float if scalar theta is provided.
    """
    if a <= 0 or d <= 0 or wavelength <= 0:
        raise ValueError("a, d, and wavelength must be positive.")
    if N < 1:
        raise ValueError("Number of slits N must be >= 1.")
    if a > d:
        raise ValueError("Slit width 'a' cannot exceed grating pitch 'd'.")

    is_scalar = np.isscalar(theta)
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=float))

    sin_diff = np.sin(theta_arr) - math.sin(theta_i)

    envelope = np.sinc((a * sin_diff) / wavelength) ** 2
    delta = (math.pi * d / wavelength) * sin_diff
    sin_delta = np.sin(delta)

    tolerance = 1e-12
    singular_mask = np.abs(sin_delta) < tolerance

    interf = np.ones_like(delta, dtype=float)
    regular_mask = ~singular_mask

    if np.any(regular_mask):
        interf[regular_mask] = (
            np.sin(N * delta[regular_mask]) / (N * sin_delta[regular_mask])
        )

    intensity = envelope * (interf**2)

    return float(intensity[0]) if is_scalar else intensity


def missing_orders(
    d: float,
    a: float,
    m_max: int = 20,
    tol: float = 1e-5,
) -> list[int]:
    """
    Determine missing diffraction orders where interference maxima coincide
    with diffraction envelope minima.

    Condition:
        m = p * (d / a) for non-zero integer p

    Parameters
    ----------
    d : float
        Grating pitch in meters.
    a : float
        Slit width in meters.
    m_max : int, optional
        Maximum positive order to test (default 20).
    tol : float, optional
        Numerical tolerance for integer checking (default 1e-5).

    Returns
    -------
    list[int]
        Sorted list of missing positive integer orders m in [1, m_max].
    """
    if a <= 0 or d <= 0:
        raise ValueError("d and a must be positive.")
    if a > d:
        raise ValueError("Slit width 'a' cannot exceed pitch 'd'.")

    ratio = d / a
    missing: list[int] = []

    p = 1
    while True:
        m_float = p * ratio
        if m_float > m_max + tol:
            break
        m_nearest = round(m_float)
        if abs(m_float - m_nearest) < tol and 1 <= m_nearest <= m_max:
            if m_nearest not in missing:
                missing.append(m_nearest)
        p += 1

    return sorted(missing)


def resolving_power(N: int, m: int) -> float:
    """
    Calculate chromatic resolving power according to Rayleigh criterion.

    Equation:
        R = lambda / Delta_lambda = |m| * N
    """
    if N < 1:
        raise ValueError("Number of rulings N must be >= 1.")
    return float(abs(m) * N)


def angular_dispersion(d: float, m: int, theta_m: float) -> float:
    """
    Calculate angular dispersion D = d(theta) / d(lambda).

    Equation:
        D = m / (d * cos(theta_m))  [radians / meter]
    """
    cos_theta = math.cos(theta_m)
    if abs(cos_theta) < 1e-12:
        raise ValueError("Angular dispersion undefined at grazing angle (cos(theta_m) ~ 0).")
    return float(m / (d * cos_theta))


def talbot_distance(d: float, wavelength: float) -> float:
    """
    Calculate Talbot self-imaging distance for near-field periodic replication.

    Equation:
        z_T = 2 * d^2 / lambda  [meters]
    """
    if d <= 0 or wavelength <= 0:
        raise ValueError("d and wavelength must be positive.")
    return float(2.0 * (d**2) / wavelength)
