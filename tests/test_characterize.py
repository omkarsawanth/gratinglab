"""Tests for gratinglab.characterize"""

import math
import pytest
import numpy as np
from scipy import special
from gratinglab import characterize
from gratinglab import constants
from gratinglab import gratings
from gratinglab import propagation


def test_synthetic_characterization_with_noise():
    np.random.seed(42)

    wavelength = constants.WAVELENGTH_RED_650
    d_true = constants.PITCH_CD
    duty_true = 0.40
    phi0_true = 1.20
    n_index = 1.50

    h_true = phi0_true * wavelength / (math.pi * (n_index - 1.0))

    grid = gratings.Grid1D(width=160.0e-6, num_samples=8192)
    a_true = duty_true * d_true
    t_clean = gratings.binary_amplitude(grid.x, d=d_true, a=a_true)

    sin_theta, intensity_clean, _ = propagation.far_field_fft(
        t_clean,
        dx=grid.dx,
        wavelength=wavelength,
        pad_factor=4,
    )

    noise = np.random.normal(0.0, 0.015, size=intensity_clean.shape)
    intensity_noisy = np.clip(intensity_clean + noise, 0.0, None)

    peak_sin, peak_heights, _ = characterize.find_orders(sin_theta, intensity_noisy, prominence_rel=0.04)

    expected_m = np.array([-2, -1, 0, 1, 2])
    matched_sin = []
    matched_heights = []

    for m in expected_m:
        target_sin = m * (wavelength / d_true)
        nearest_idx = np.argmin(np.abs(peak_sin - target_sin))
        matched_sin.append(peak_sin[nearest_idx])
        matched_heights.append(peak_heights[nearest_idx])

    matched_sin = np.array(matched_sin)
    matched_heights = np.array(matched_heights)

    d_recovered, _ = characterize.estimate_period(matched_sin, expected_m, wavelength)
    d_rel_error_pct = abs(d_recovered - d_true) / d_true * 100.0
    assert d_rel_error_pct < 2.0

    m_duty_orders = np.array([-3, -2, -1, 0, 1, 2, 3])
    noisy_order_heights = np.sinc(m_duty_orders * duty_true) ** 2 + np.random.normal(0.0, 0.010, size=m_duty_orders.shape)
    duty_recovered, _ = characterize.estimate_duty_cycle(noisy_order_heights, m_duty_orders)
    duty_rel_error_pct = abs(duty_recovered - duty_true) / duty_true * 100.0
    assert duty_rel_error_pct < 2.0

    noisy_eff = special.jv(expected_m, phi0_true) ** 2 + np.random.normal(0.0, 0.012, size=expected_m.shape)
    h_recovered, _, _ = characterize.estimate_groove_depth(
        noisy_eff,
        expected_m,
        wavelength=wavelength,
        n_index=n_index,
    )
    h_rel_error_pct = abs(h_recovered - h_true) / h_true * 100.0
    assert h_rel_error_pct < 2.0


def test_lattice_constants_2d():
    wavelength = 650.0e-9
    dx_true = 2.0e-6
    dy_true = 3.0e-6

    peaks = [
        (0.0, 0.0),
        (wavelength / dx_true, 0.0),
        (-wavelength / dx_true, 0.0),
        (0.0, wavelength / dy_true),
        (0.0, -wavelength / dy_true),
    ]

    dx_rec, dy_rec = characterize.lattice_constants_2d(peaks, wavelength)
    assert pytest.approx(dx_true, rel=1e-5) == dx_rec
    assert pytest.approx(dy_true, rel=1e-5) == dy_rec
