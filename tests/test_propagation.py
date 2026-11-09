"""Tests for gratinglab.propagation"""

import math
import pytest
import numpy as np
from scipy.signal import find_peaks
from gratinglab import analytical
from gratinglab import constants
from gratinglab import gratings
from gratinglab import propagation


def test_gaussian_beam_profiles():
    w0 = 50.0e-6
    grid = gratings.Grid1D(width=400.0e-6, num_samples=2000)

    beam_1d = propagation.gaussian_beam(grid.x, w0=w0)
    assert pytest.approx(1.0, abs=1e-12) == np.abs(beam_1d[grid.num_samples // 2])

    grid_2d = gratings.Grid2D(width_x=200.0e-6, num_samples_x=100)
    beam_2d = propagation.gaussian_beam_2d(grid_2d.grid_x.x, grid_2d.grid_y.x, w0_x=w0)
    assert beam_2d.shape == (100, 100)
    assert pytest.approx(1.0, abs=1e-12) == np.abs(beam_2d[50, 50])


def test_fft_peak_positions_match_analytical():
    wavelength = constants.WAVELENGTH_RED_650
    d = constants.PITCH_CD
    a = 0.5e-6

    width = 160.0e-6
    num_samples = 8192
    grid = gratings.Grid1D(width=width, num_samples=num_samples)

    t_grating = gratings.binary_amplitude(grid.x, d=d, a=a)

    sin_theta, intensity, _ = propagation.far_field_fft(
        t_grating,
        dx=grid.dx,
        wavelength=wavelength,
        pad_factor=4,
    )

    peaks, _ = find_peaks(intensity, height=0.05, distance=50)
    peak_sin_values = sin_theta[peaks]

    expected_orders, _ = analytical.order_angles(d=d, wavelength=wavelength, m_max=2)
    expected_sin_1 = math.sin(expected_orders[1])

    pos_peaks = peak_sin_values[peak_sin_values > 0.1]
    measured_sin_1 = pos_peaks[np.argmin(np.abs(pos_peaks - expected_sin_1))]

    rel_error = abs(measured_sin_1 - expected_sin_1) / expected_sin_1
    assert rel_error < 0.01


def test_parseval_energy_conservation():
    wavelength = 650.0e-9
    d = 2.0e-6
    width = 40.0e-6
    num_samples = 2048
    grid = gratings.Grid1D(width=width, num_samples=num_samples)

    t_grating = gratings.binary_amplitude(grid.x, d=d, a=1.0e-6)
    beam = propagation.gaussian_beam(grid.x, w0=12.0e-6)
    field_0 = t_grating * beam

    energy_0 = np.sum(np.abs(field_0) ** 2) * grid.dx

    field_z = propagation.angular_spectrum(
        field_0,
        dx=grid.dx,
        wavelength=wavelength,
        z=10.0e-6,
    )
    energy_z = np.sum(np.abs(field_z) ** 2) * grid.dx

    assert pytest.approx(energy_0, rel=0.06) == energy_z


def test_talbot_self_imaging():
    wavelength = 650.0e-9
    d = 1.6e-6
    a = 0.8e-6

    width = 160.0e-6
    num_samples = 8192
    grid = gratings.Grid1D(width=width, num_samples=num_samples)

    t_grating = gratings.binary_amplitude(grid.x, d=d, a=a)
    z_talbot = analytical.talbot_distance(d=d, wavelength=wavelength)

    u_talbot = propagation.angular_spectrum(
        t_grating,
        dx=grid.dx,
        wavelength=wavelength,
        z=z_talbot,
    )

    intensity_0 = np.abs(t_grating) ** 2
    intensity_zt = np.abs(u_talbot) ** 2

    center_slice = slice(num_samples // 4, 3 * (num_samples // 4))
    i0_center = intensity_0[center_slice]
    iz_center = intensity_zt[center_slice]

    correlation = np.corrcoef(i0_center, iz_center)[0, 1]
    assert correlation > 0.85
