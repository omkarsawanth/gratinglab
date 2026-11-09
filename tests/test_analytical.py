"""Tests for gratinglab.analytical"""

import math
import pytest
from gratinglab import analytical


def test_cd_pitch_first_order_angle():
    wavelength = 650.0e-9
    d_cd = 1.6e-6
    orders, evanescent = analytical.order_angles(d=d_cd, wavelength=wavelength)

    assert 1 in orders
    theta_1_deg = math.degrees(orders[1])
    assert pytest.approx(23.976, abs=0.05) == theta_1_deg

    assert -1 in orders
    assert pytest.approx(-orders[1]) == orders[-1]

    assert 3 in evanescent
    assert -3 in evanescent


def test_dvd_pitch_evanescent_order():
    wavelength = 650.0e-9
    d_dvd = 0.74e-6
    orders, evanescent = analytical.order_angles(d=d_dvd, wavelength=wavelength)

    assert 1 in orders
    theta_1_deg = math.degrees(orders[1])
    assert pytest.approx(61.4, abs=0.1) == theta_1_deg

    assert 2 in evanescent
    assert -2 in evanescent
    assert 2 not in orders


def test_missing_orders_even():
    d = 2.0e-6
    a = 1.0e-6
    missing = analytical.missing_orders(d=d, a=a, m_max=10)
    assert missing == [2, 4, 6, 8, 10]

    wavelength = 650.0e-9
    orders, _ = analytical.order_angles(d=d, wavelength=wavelength)
    theta_2 = orders[2]

    intensity_at_order_2 = analytical.n_slit_intensity(
        theta=theta_2,
        a=a,
        d=d,
        N=10,
        wavelength=wavelength,
    )
    assert pytest.approx(0.0, abs=1e-12) == intensity_at_order_2


def test_n_slit_intensity_peak_limit():
    wavelength = 650.0e-9
    d = 1.6e-6
    a = 0.4e-6

    i_center = analytical.n_slit_intensity(0.0, a, d, N=20, wavelength=wavelength)
    assert pytest.approx(1.0, abs=1e-12) == i_center


def test_resolving_power_and_dispersion():
    N = 600
    m = 2
    assert analytical.resolving_power(N, m) == 1200.0

    d = 1.6e-6
    wavelength = 650.0e-9
    orders, _ = analytical.order_angles(d, wavelength)
    theta_1 = orders[1]

    D = analytical.angular_dispersion(d=d, m=1, theta_m=theta_1)
    assert pytest.approx(683977.8, rel=1e-3) == D


def test_talbot_distance():
    d = 1.6e-6
    wavelength = 650.0e-9
    z_t = analytical.talbot_distance(d, wavelength)
    assert pytest.approx(7.8769e-6, rel=1e-3) == z_t
