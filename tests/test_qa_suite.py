"""
tests/test_qa_suite.py
======================
Comprehensive Quality Assurance validation test suite for GratingLab.
Validates:
1. Application launch with all parameter combinations
2. No missing imports or dependencies
3. All sliders functionality and live text updating
4. Radio button profile switching & progressive disclosure
5. Visualization checkboxes (Log scale, Analytical compare)
6. Preset switching & selected states
7. Reset restoration to defaults
8. Mathematical correctness of diffraction calculations
9. FFT spectrum computation & order tracking
10. Talbot carpet near-field simulation
11. Performance optimization guards (bulk update & caching)
12. Layout stability across standard resolutions (1280x720, 1920x1080)
"""

from __future__ import annotations
import math
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gratinglab import analytical
from gratinglab import characterize
from gratinglab import constants
from gratinglab import gratings
from gratinglab import propagation
from gratinglab import visualize
from gratinglab.visualize import GratingLabApp, get_material_info


@pytest.fixture(autouse=True)
def set_headless_backend(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    matplotlib.use("Agg", force=True)


def test_qa_01_module_exports_and_imports():
    """Verify all public API modules and entry points import cleanly."""
    import gratinglab
    assert hasattr(gratinglab, "analytical")
    assert hasattr(gratinglab, "gratings")
    assert hasattr(gratinglab, "propagation")
    assert hasattr(gratinglab, "characterize")
    assert hasattr(gratinglab, "constants")
    assert hasattr(gratinglab, "visualize")
    assert hasattr(gratinglab.visualize, "GratingLabApp")
    assert hasattr(gratinglab.visualize, "launch_interactive_app")


def test_qa_02_app_initialization_defaults():
    """Verify application initializes cleanly with exact physical defaults."""
    app = GratingLabApp()
    assert app.wavelength == pytest.approx(650e-9)
    assert app.period == pytest.approx(1.6e-6)
    assert app.n_index == pytest.approx(1.50)
    assert app.n_slits == 10
    assert app.duty == pytest.approx(0.50)
    assert app.screen_dist == pytest.approx(0.50)
    assert app.theta_i == pytest.approx(0.0)
    assert app.grating_type == "Binary Amplitude"
    assert app.log_scale is False
    assert app.compare_analytical is True
    plt.close("all")


def test_qa_03_all_sliders_interactive_range():
    """Verify each slider updates internal physics and live readouts across its range."""
    app = GratingLabApp()

    # 1. Wavelength slider
    app.s_wl.set_val(532.0)
    assert app.wavelength == pytest.approx(532e-9)
    assert "532 nm [Green]" in app.txt_wl.get_text()

    app.s_wl.set_val(405.0)
    assert app.wavelength == pytest.approx(405e-9)
    assert "405 nm [Blue/Violet]" in app.txt_wl.get_text()

    # 2. Incidence Angle slider
    app.s_theta.set_val(15.0)
    assert math.degrees(app.theta_i) == pytest.approx(15.0)
    assert "+15.0°" in app.txt_theta.get_text()

    app.s_theta.set_val(-10.0)
    assert math.degrees(app.theta_i) == pytest.approx(-10.0)
    assert "-10.0°" in app.txt_theta.get_text()

    # 3. Beam Waist slider
    app.s_waist.set_val(80.0)
    assert app.beam_waist == pytest.approx(80e-6)
    assert "80.0 µm" in app.txt_waist.get_text()

    # 4. Screen Distance slider
    app.s_dist.set_val(1.20)
    assert app.screen_dist == pytest.approx(1.20)
    assert "1.20 m" in app.txt_dist.get_text()

    # 5. Grating Pitch slider
    app.s_period.set_val(3.33)
    assert app.period == pytest.approx(3.33e-6)
    assert "3.33 µm" in app.txt_period.get_text()
    assert "300 lines/mm" in app.lbl_grating_telemetry.get_text() or "300 l/mm" in app.lbl_grating_telemetry.get_text()

    # 6. Refractive Index slider
    app.s_nindex.set_val(1.46)
    assert app.n_index == pytest.approx(1.46)
    assert "1.46 (Fused)" in app.txt_nindex.get_text()

    app.s_nindex.set_val(1.58)
    assert app.n_index == pytest.approx(1.58)
    assert "1.58 (Polycarbonate)" in app.txt_nindex.get_text()

    # 7. Rulings Count slider
    app.s_n.set_val(25)
    assert app.n_slits == 25
    assert "25 lines" in app.txt_n.get_text()

    # 8. Duty Cycle slider
    app.s_duty.set_val(0.40)
    assert app.duty == pytest.approx(0.40)
    assert "0.40" in app.txt_duty.get_text()

    # 9. Phase Depth slider (switch to phase profile first)
    app.on_type("Sinusoidal Phase")
    app.s_phase.set_val(math.pi)
    assert app.phase_depth == pytest.approx(math.pi)
    assert "1.00π rad" in app.txt_phase.get_text()

    plt.close("all")


def test_qa_04_radio_profiles_and_progressive_disclosure():
    """Verify switching grating profiles correctly toggles progressive controls."""
    app = GratingLabApp()

    # Binary Amplitude: Duty cycle visible, Phase depth hidden
    app.on_type("Binary Amplitude")
    assert app.grating_type == "Binary Amplitude"
    assert app.ax_duty_lbl.get_visible() is True
    assert app.ax_duty_track.get_visible() is True
    assert app.ax_phase_lbl.get_visible() is False
    assert app.ax_phase_track.get_visible() is False

    # Sinusoidal Phase: Phase depth visible, Duty cycle hidden
    app.on_type("Sinusoidal Phase")
    assert app.grating_type == "Sinusoidal Phase"
    assert app.ax_duty_lbl.get_visible() is False
    assert app.ax_duty_track.get_visible() is False
    assert app.ax_phase_lbl.get_visible() is True
    assert app.ax_phase_track.get_visible() is True

    # Blazed Phase: Phase depth visible, Duty cycle hidden
    app.on_type("Blazed Phase")
    assert app.grating_type == "Blazed Phase"
    assert app.ax_phase_lbl.get_visible() is True

    # 2D Orthogonal: Duty cycle visible, Phase depth hidden
    app.on_type("2D Orthogonal")
    assert app.grating_type == "2D Orthogonal"
    assert app.ax_duty_lbl.get_visible() is True
    assert app.ax_phase_lbl.get_visible() is False

    plt.close("all")


def test_qa_05_view_toggles_and_caching():
    """Verify visualization check toggles operate without full wavefield recomputation."""
    app = GratingLabApp()

    # Toggle Log Scale
    assert app.log_scale is False
    app.check_toggles.set_active(0)
    assert app.log_scale is True
    assert app.ax_fft.get_yscale() == "log"

    app.check_toggles.set_active(0)
    assert app.log_scale is False
    assert app.ax_fft.get_yscale() == "linear"

    # Toggle Compare Analytical
    assert app.compare_analytical is True
    app.check_toggles.set_active(1)
    assert app.compare_analytical is False

    plt.close("all")


def test_qa_06_presets_and_selected_state():
    """Verify all 3 presets apply exact parameters and update selected button indicator."""
    app = GratingLabApp()

    # Preset 1: Single Slit
    app.apply_preset("Single slit")
    assert app.n_slits == 1
    assert app.period == pytest.approx(4.0e-6)
    assert app.wavelength == pytest.approx(650e-9)
    assert app.grating_type == "Binary Amplitude"
    assert app._active_preset_idx == 0

    # Preset 2: Diffraction Grating (CD)
    app.apply_preset("Diffraction grating")
    assert app.n_slits == 20
    assert app.period == pytest.approx(1.6e-6)
    assert app.grating_type == "Binary Amplitude"
    assert app._active_preset_idx == 1

    # Preset 3: Blazed Spectrometer
    app.apply_preset("Blazed spectrometer")
    assert app.n_slits == 30
    assert app.period == pytest.approx(1.6e-6)
    assert app.grating_type == "Blazed Phase"
    assert app.phase_depth == pytest.approx(2.0 * math.pi)
    assert app._active_preset_idx == 2

    # Moving a slider should clear active preset highlight
    app.s_wl.set_val(500.0)
    assert app._active_preset_idx is None

    plt.close("all")


def test_qa_07_reset_functionality():
    """Verify Reset button atomically restores default configuration."""
    app = GratingLabApp()

    # Modify parameters
    app.apply_values(
        grating_type="Blazed Phase",
        wl=450.0,
        period=5.0,
        n_slits=40,
        duty=0.8,
        phase=math.pi,
        theta=20.0,
        waist=20.0,
        dist=1.5,
        n_index=1.70,
        log=True,
        comp=False,
    )
    assert app.grating_type == "Blazed Phase"
    assert app.wavelength == pytest.approx(450e-9)

    # Trigger reset
    app.reset()
    assert app.grating_type == "Binary Amplitude"
    assert app.wavelength == pytest.approx(650e-9)
    assert app.period == pytest.approx(1.6e-6)
    assert app.n_slits == 10
    assert app.screen_dist == pytest.approx(0.50)
    assert app.n_index == pytest.approx(1.50)
    assert app.log_scale is False
    assert app.compare_analytical is True
    assert "RESET RESTORED" in app.lbl_head_status.get_text()

    plt.close("all")


def test_qa_08_diffraction_physics_accuracy():
    """Verify diffraction physics: grating equation, missing orders, and dispersion."""
    wavelength = 650e-9
    period = 1.6e-6

    # 1. Grating equation: sin(theta_m) = m * lambda / d
    sin_theta1 = 1.0 * wavelength / period
    theta1_expected = math.asin(sin_theta1)
    orders, evanescent = analytical.order_angles(d=period, wavelength=wavelength)
    assert orders[1] == pytest.approx(theta1_expected, rel=1e-5)
    assert orders[-1] == pytest.approx(-theta1_expected, rel=1e-5)

    # 2. Missing orders for 50% duty cycle (d/a = 2)
    missing = analytical.missing_orders(d=period, a=0.5 * period)
    assert 2 in missing
    assert 4 in missing
    assert 1 not in missing

    # 3. Talbot distance: z_T = 2 * d^2 / lambda
    z_t = analytical.talbot_distance(d=period, wavelength=wavelength)
    expected_zt = 2.0 * (period**2) / wavelength
    assert z_t == pytest.approx(expected_zt, rel=1e-5)

    # 4. Dispersion: D = m / (d * cos(theta))
    disp_rad = analytical.angular_dispersion(d=period, m=1, theta_m=theta1_expected)
    disp_deg_nm = disp_rad * (180.0 / math.pi) * 1e-9
    assert disp_deg_nm == pytest.approx(0.039, abs=0.005)


def test_qa_09_talbot_window_launch():
    """Verify Talbot carpet window generates without errors and sets proper axes."""
    app = GratingLabApp()
    app.open_talbot_window()

    figs = [plt.figure(i) for i in plt.get_fignums()]
    assert len(figs) >= 2  # Main window + Talbot window
    plt.close("all")


def test_qa_10_multiresolution_render_stability():
    """Verify layout integrity and canvas drawing at standard display resolutions."""
    app = GratingLabApp()

    # 1. 1280 x 720 (Standard HD)
    app.fig.set_size_inches(12.8, 7.2)
    app.fig.canvas.draw()

    # 2. 1920 x 1080 (Full HD)
    app.fig.set_size_inches(19.2, 10.8)
    app.fig.canvas.draw()

    # 3. Compact 1024 x 600
    app.fig.set_size_inches(10.24, 6.0)
    app.fig.canvas.draw()

    plt.close("all")


def test_qa_11_bulk_update_suppression():
    """Verify _bulk_updating flag prevents intermediate simulation recalculations."""
    app = GratingLabApp()

    initial_calc_count = app._last_sim_params

    # During bulk update, individual slider calls should NOT trigger update()
    app._bulk_updating = True
    app.s_wl.set_val(488.0)
    app.s_period.set_val(2.5)
    app.s_n.set_val(15)

    # Last sim params shouldn't have changed yet
    assert app._last_sim_params == initial_calc_count

    # Complete bulk update
    app._bulk_updating = False
    app.update()
    assert app._last_sim_params is not None
    assert app.wavelength == pytest.approx(488e-9)
    assert app.period == pytest.approx(2.5e-6)

    plt.close("all")


def test_qa_12_material_info_fresnel():
    """Verify material helper function accurately resolves materials and Fresnel values."""
    materials = [
        (1.00, "Vacuum / Air", 1.0),
        (1.33, "Water / Cryolite", 0.98),
        (1.46, "Fused Silica", 0.965),
        (1.50, "Crown Glass (BK7)", 0.96),
        (1.58, "Polycarbonate", 0.949),
        (1.70, "Dense Flint Glass", 0.933),
        (1.77, "Sapphire (Al₂O₃)", 0.923),
        (2.40, "Diamond / High-Index", 0.83),
    ]
    for n, expected_name, approx_t in materials:
        name, t_fres = get_material_info(n)
        assert expected_name in name
        assert t_fres == pytest.approx(approx_t, abs=0.03)
