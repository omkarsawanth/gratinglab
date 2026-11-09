"""Tests for gratinglab.visualize"""

import pytest
from gratinglab import visualize


def test_wavelength_to_rgb():
    r, g, b = visualize.wavelength_to_rgb(650.0)
    assert r > 0.8 and g == 0.0 and b == 0.0

    r, g, b = visualize.wavelength_to_rgb(532.0)
    assert r < 0.4 and g > 0.8 and b == 0.0

    r, g, b = visualize.wavelength_to_rgb(405.0)
    assert r > 0.3 and g == 0.0 and b > 0.6


def test_get_laser_colormap():
    cmap = visualize.get_laser_colormap(650.0)
    assert cmap is not None
    c_zero = cmap(0.0)
    assert c_zero[0] == 0.0 and c_zero[1] == 0.0 and c_zero[2] == 0.0


def test_get_material_info():
    name, t_fresnel = visualize.get_material_info(1.0)
    assert "Vacuum" in name
    assert t_fresnel == 1.0

    name_silica, t_silica = visualize.get_material_info(1.46)
    assert "Silica" in name_silica
    assert 0.95 <= t_silica <= 0.98

    name_bk7, t_bk7 = visualize.get_material_info(1.50)
    assert "Glass" in name_bk7
    assert 0.95 <= t_bk7 <= 0.97


def test_app_initialization_and_interaction(monkeypatch):
    import os
    monkeypatch.setenv("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    app = visualize.GratingLabApp()
    assert app.wavelength == pytest.approx(650e-9)
    assert app.period == pytest.approx(1.6e-6)
    assert app.n_index == pytest.approx(1.50)
    assert app.grating_type == "Binary Amplitude"
    assert app.s_nindex.val == pytest.approx(1.50)

    # Test Presets
    app.apply_preset("Single slit")
    assert app.n_slits == 1
    assert app.period == pytest.approx(4.0e-6)
    assert app._active_preset_idx == 0

    app.apply_preset("Blazed spectrometer")
    assert app.grating_type == "Blazed Phase"
    assert app._active_preset_idx == 2

    # Test Reset
    app.reset()
    assert app.wavelength == pytest.approx(650e-9)
    assert app.period == pytest.approx(1.6e-6)
    assert app.n_slits == 10
    assert app.grating_type == "Binary Amplitude"
    assert app._active_preset_idx is None

    # Test View toggles (optimization check)
    app.check_toggles.set_active(0)  # Toggle Log Scale
    assert app.log_scale is True

    plt.close("all")
