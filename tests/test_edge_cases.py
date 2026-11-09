"""
tests/test_edge_cases.py
========================
Edge case test suite for GratingLab:
1. Evanescent orders: deep sub-wavelength pitches (d < lambda) and grazing diffraction.
2. Missing orders: rational ratio d/a = 1.5 (orders 3, 6, 9 vanish).
3. Under-sampled and narrow computational grids: asserts UserWarning triggers.
4. Asymptotically large N (N = 1000, 5000): numerical stability, zero NaN/Inf,
   FWHM scaling inversely with N.
"""

import math
import warnings
import pytest
import numpy as np
from gratinglab import analytical
from gratinglab import constants
from gratinglab import gratings


def test_evanescent_orders_deep_subwavelength():
    d_sub = 0.30e-6
    wavelength = 650.0e-9

    orders, evanescent = analytical.order_angles(d=d_sub, wavelength=wavelength, m_max=3)

    assert list(orders.keys()) == [0]
    assert orders[0] == 0.0

    for m in [-3, -2, -1, 1, 2, 3]:
        assert m in evanescent
        assert m not in orders


def test_grazing_diffraction_angle():
    wavelength = 500.0e-9
    d = 500.0e-9

    orders, _ = analytical.order_angles(d=d, wavelength=wavelength, m_max=1)
    assert 1 in orders
    assert pytest.approx(math.pi / 2.0, abs=1e-7) == orders[1]

    with pytest.raises(ValueError, match="grazing angle"):
        analytical.angular_dispersion(d=d, m=1, theta_m=orders[1])


def test_missing_orders_rational_ratio():
    d = 3.0e-6
    a = 2.0e-6
    wavelength = 650.0e-9

    missing = analytical.missing_orders(d=d, a=a, m_max=8)
    assert 3 in missing
    assert 6 in missing
    assert 1 not in missing
    assert 2 not in missing
    assert 4 not in missing

    orders, _ = analytical.order_angles(d=d, wavelength=wavelength, m_max=3)
    theta_3 = orders[3]
    intensity_at_order_3 = analytical.n_slit_intensity(
        theta=theta_3,
        a=a,
        d=d,
        N=10,
        wavelength=wavelength,
    )
    assert pytest.approx(0.0, abs=1e-12) == intensity_at_order_3


def test_undersampled_and_narrow_grid_warnings():
    d = 1.6e-6

    grid_coarse = gratings.Grid1D(width=20.0e-6, num_samples=100)
    with pytest.warns(UserWarning, match="under-resolved"):
        grid_coarse.check_resolution(d)

    grid_narrow = gratings.Grid1D(width=8.0e-6, num_samples=1000)
    with pytest.warns(UserWarning, match="fewer than 10"):
        grid_narrow.check_resolution(d)


def test_asymptotically_large_slit_counts():
    wavelength = 650.0e-9
    d = 1.6e-6
    a = 0.4e-6

    theta_dense = np.linspace(-0.02, 0.02, 50000)

    i_1000 = analytical.n_slit_intensity(theta_dense, a, d, N=1000, wavelength=wavelength)
    assert not np.any(np.isnan(i_1000))
    assert not np.any(np.isinf(i_1000))
    assert pytest.approx(1.0, abs=1e-4) == np.max(i_1000)

    i_5000 = analytical.n_slit_intensity(theta_dense, a, d, N=5000, wavelength=wavelength)
    assert not np.any(np.isnan(i_5000))
    assert not np.any(np.isinf(i_5000))
    assert pytest.approx(1.0, abs=1e-4) == np.max(i_5000)

    fwhm_1000_idx = np.sum(i_1000 >= 0.5)
    fwhm_5000_idx = np.sum(i_5000 >= 0.5)

    width_ratio = fwhm_1000_idx / fwhm_5000_idx
    assert pytest.approx(5.0, rel=0.08) == width_ratio
