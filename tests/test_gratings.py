"""Tests for gratinglab.gratings"""

import math
import warnings
import pytest
import numpy as np
from gratinglab import gratings


def test_grid_resolution_warnings():
    d = 1.0e-6

    grid_under_resolved = gratings.Grid1D(width=20.0e-6, num_samples=200)
    with pytest.warns(UserWarning, match="under-resolved"):
        grid_under_resolved.check_resolution(d)

    grid_short_window = gratings.Grid1D(width=5.0e-6, num_samples=2000)
    with pytest.warns(UserWarning, match="fewer than 10"):
        grid_short_window.check_resolution(d)

    grid_good = gratings.Grid1D(width=20.0e-6, num_samples=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        grid_good.check_resolution(d)


def test_periodicity():
    d = 1.6e-6
    x = np.linspace(-3.0 * d, 3.0 * d, 1000)

    t_amp = gratings.binary_amplitude(x, d=d, a=0.8e-6)
    t_amp_shifted = gratings.binary_amplitude(x + d, d=d, a=0.8e-6)
    np.testing.assert_allclose(t_amp, t_amp_shifted, atol=1e-12)

    t_sin = gratings.sinusoidal_phase(x, d=d, phi0=math.pi / 2.0)
    t_sin_shifted = gratings.sinusoidal_phase(x + d, d=d, phi0=math.pi / 2.0)
    np.testing.assert_allclose(t_sin, t_sin_shifted, atol=1e-12)

    t_blaze = gratings.blazed(x, d=d, phase_depth=2.0 * math.pi)
    t_blaze_shifted = gratings.blazed(x + d, d=d, phase_depth=2.0 * math.pi)
    np.testing.assert_allclose(t_blaze, t_blaze_shifted, atol=1e-12)

    t_bin_phase = gratings.binary_phase(x, d=d, duty=0.5, phi=math.pi)
    t_bin_phase_shifted = gratings.binary_phase(x + d, d=d, duty=0.5, phi=math.pi)
    np.testing.assert_allclose(t_bin_phase, t_bin_phase_shifted, atol=1e-12)


def test_amplitude_gratings_bounded():
    d = 1.6e-6
    x = np.linspace(-5.0 * d, 5.0 * d, 2000)
    t_amp = gratings.binary_amplitude(x, d=d, a=0.6e-6)
    magnitudes = np.abs(t_amp)
    assert np.all(magnitudes <= 1.0)


def test_phase_gratings_unit_magnitude():
    d = 0.74e-6
    x = np.linspace(-5.0 * d, 5.0 * d, 2000)
    t_sin = gratings.sinusoidal_phase(x, d=d, phi0=1.2)
    t_blaze = gratings.blazed(x, d=d, phase_depth=2.0 * math.pi)
    t_bin = gratings.binary_phase(x, d=d, duty=0.4, phi=math.pi)

    np.testing.assert_allclose(np.abs(t_sin), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(t_blaze), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(t_bin), 1.0, atol=1e-12)


def test_duty_cycle_measurement():
    d = 1.0e-6
    duty_requested = 0.35
    grid = gratings.Grid1D(width=100.0 * d, num_samples=50000)
    a = duty_requested * d
    t_amp = gratings.binary_amplitude(grid.x, d=d, a=a)
    measured_duty_amp = np.mean(np.abs(t_amp) > 0.5)
    assert pytest.approx(duty_requested, abs=2.0 * (grid.dx / d)) == measured_duty_amp


def test_finite_aperture_line_count():
    d = 1.6e-6
    n_lines = 10
    grid = gratings.Grid1D(width=30.0 * d, num_samples=3000)
    mask = gratings.finite_aperture(grid.x, n_lines=n_lines, d=d)
    illuminated_width = np.sum(mask) * grid.dx
    expected_width = n_lines * d
    assert pytest.approx(expected_width, abs=2.0 * grid.dx) == illuminated_width


def test_grating_2d_shapes_and_values():
    grid = gratings.Grid2D(width_x=10.0e-6, num_samples_x=100)
    t_2d_rect = gratings.grating_2d(grid.grid_x.x, grid.grid_y.x, dx_period=2.0e-6, dy_period=2.0e-6, kind="rect_aperture")
    assert t_2d_rect.shape == (100, 100)
    assert np.all(np.abs(t_2d_rect) <= 1.0)
