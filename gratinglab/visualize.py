"""
gratinglab.visualize
====================
Research-grade scientific visualization and instrument dashboard for optical grating
diffraction, Angular Spectrum propagation, and material metrology.
Inspired by laboratory spectrum analyzers, Bloomberg-grade data density, and modern dark CAD metrology.

Visualizations:
1. Aperture Plane Transmittance: Field magnitude |t(x)| area fill, phase shift φ(x)/π,
   illuminating Gaussian beam envelope w₀, and period pitch (d) dimension annotation.
2. Far-Field Fraunhofer Diffraction: High-resolution FFT spectrum with area fill,
   analytical N-slit model, single-slit envelope sinc²(β), diffraction peak maximum points,
   suppressed missing orders, and blaze order highlighting.
3. Calibrated Optical Detector Array: Physically divergent 2D laser spots, calibrated
   transverse metric scale y = L·tan(θ) in cm, and downward spot pointer reticles.
4. Near-Field Talbot Carpet: Multi-plane fractional Talbot self-imaging (z_T, z_T/2, z_T/4),
   dual physical (µm) and normalized (z/z_T) coordinates, and calibrated wavefield colorbar.
"""

from __future__ import annotations
import math
import os
import sys

# Ensure an interactive backend is used for desktop GUI rendering if not in headless mode
import matplotlib
if os.environ.get("MPLBACKEND", "").lower() != "agg" and matplotlib.get_backend().lower() in ("agg", "template", "pdf", "ps", "svg"):
    for _backend in ["QtAgg", "TkAgg", "Qt5Agg", "WXAgg"]:
        try:
            import importlib
            # Check if backend module can be imported
            if _backend == "QtAgg":
                import PyQt6
            elif _backend == "TkAgg":
                import tkinter
            elif _backend == "Qt5Agg":
                import PyQt5
            elif _backend == "WXAgg":
                import wx
            matplotlib.use(_backend, force=True)
            break
        except Exception:
            continue

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider
import numpy as np
from gratinglab import analytical
from gratinglab import constants
from gratinglab import gratings
from gratinglab import propagation


def wavelength_to_rgb(wavelength_nm: float) -> tuple[float, float, float]:
    wl = float(wavelength_nm)
    if 380 <= wl < 440:
        r = -(wl - 440.0) / (440.0 - 380.0)
        g = 0.0
        b = 1.0
    elif 440 <= wl < 490:
        r = 0.0
        g = (wl - 440.0) / (490.0 - 440.0)
        b = 1.0
    elif 490 <= wl < 510:
        r = 0.0
        g = 1.0
        b = -(wl - 510.0) / (510.0 - 490.0)
    elif 510 <= wl < 580:
        r = (wl - 510.0) / (580.0 - 510.0)
        g = 1.0
        b = 0.0
    elif 580 <= wl < 645:
        r = 1.0
        g = -(wl - 645.0) / (645.0 - 580.0)
        b = 0.0
    elif 645 <= wl <= 750:
        r = 1.0
        g = 0.0
        b = 0.0
    else:
        r, g, b = 0.6, 0.6, 0.6

    if 380 <= wl < 420:
        factor = 0.3 + 0.7 * (wl - 380.0) / (420.0 - 380.0)
    elif 420 <= wl < 700:
        factor = 1.0
    elif 700 <= wl <= 750:
        factor = 0.3 + 0.7 * (750.0 - wl) / (750.0 - 700.0)
    else:
        factor = 0.2

    return (
        max(0.0, min(1.0, r * factor)),
        max(0.0, min(1.0, g * factor)),
        max(0.0, min(1.0, b * factor)),
    )


def get_laser_colormap(wavelength_nm: float) -> LinearSegmentedColormap:
    r, g, b = wavelength_to_rgb(wavelength_nm)
    laser_color = (r, g, b)
    colors = [
        (0.0, 0.0, 0.0),
        (0.2 * r, 0.2 * g, 0.2 * b),
        laser_color,
        (1.0, 1.0, 1.0),
    ]
    return LinearSegmentedColormap.from_list(f"laser_{int(wavelength_nm)}", colors, N=256)


class GratingLabApp:
    def __init__(
        self,
        grating_type: str = "Binary Amplitude",
        wavelength_nm: float = 650.0,
        period_um: float = 1.6,
        n_slits: int = 10,
        screen_dist_m: float = 0.5,
    ) -> None:
        # Default snapshot for the Reset button
        self.default_params = {
            "grating_type": grating_type,
            "wavelength_nm": wavelength_nm,
            "period_um": period_um,
            "n_slits": n_slits,
            "screen_dist_m": screen_dist_m,
            "duty": 0.5,
            "theta_i_deg": 0.0,
            "beam_waist_um": 40.0,
            "phase_depth": 2.0 * math.pi,
            "log_scale": False,
            "compare_analytical": True,
        }

        # Operational state
        self.grating_type = grating_type
        self.wavelength = wavelength_nm * 1e-9
        self.period = period_um * 1e-6
        self.duty = 0.5
        self.n_slits = n_slits
        self.theta_i = 0.0
        self.screen_dist = screen_dist_m
        self.beam_waist = 40.0e-6
        self.phase_depth = 2.0 * math.pi
        self.log_scale = False
        self.compare_analytical = True
        self.num_samples = 2048

        self.build_layout()
        self.build_controls()
        self.on_type(self.grating_type)

    def build_layout(self) -> None:
        """Create pure GridSpec dark scientific dashboard architecture without manual coordinates."""
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(16, 9.5), dpi=100, facecolor="#0b0f14")
        self.fig.canvas.manager.set_window_title("GratingLab // Precision Optical Metrology & Diffraction Studio")

        # 4 Primary Vertical Tiers:
        # Tier 0: Instrument Navigation & Quick Preset Bar
        # Tier 1: Dual Oscilloscope Simulation Workspace
        # Tier 2: Calibrated Optical Detector Array (Screen Strip)
        # Tier 3: 4-Column Control Deck & High-Density Telemetry Terminal
        self.gs_main = self.fig.add_gridspec(
            4, 1,
            height_ratios=[0.38, 3.2, 0.65, 3.75],
            hspace=0.46,
            left=0.035, right=0.965, top=0.965, bottom=0.035,
        )

        # ─── Top Instrument Header Bar ───
        gs_head = self.gs_main[0].subgridspec(1, 3, width_ratios=[1.3, 0.9, 1.4], wspace=0.1)
        self.ax_head_brand = self.fig.add_subplot(gs_head[0])
        self.ax_head_brand.axis("off")

        self.ax_head_status = self.fig.add_subplot(gs_head[1])
        self.ax_head_status.axis("off")

        self.gs_hbtns = gs_head[2].subgridspec(1, 5, wspace=0.12)

        # ─── Dual-Channel Oscilloscopes ───
        gs_plots = self.gs_main[1].subgridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.16)
        self.ax_profile = self.fig.add_subplot(gs_plots[0], facecolor="#11161d")
        self.ax_fft = self.fig.add_subplot(gs_plots[1], facecolor="#11161d")

        for ax in (self.ax_profile, self.ax_fft):
            ax.tick_params(colors="#8b949e", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#242c38")
                spine.set_linewidth(0.8)

        # ─── Detector Array Screen Strip ───
        self.ax_screen = self.fig.add_subplot(self.gs_main[2], facecolor="#000000")
        self.ax_screen.tick_params(colors="#8b949e", labelsize=8)
        for spine in self.ax_screen.spines.values():
            spine.set_color("#242c38")
            spine.set_linewidth(0.8)

        # ─── 4-Column Control Deck & Terminal ───
        self.gs_controls = self.gs_main[3].subgridspec(
            1, 4,
            width_ratios=[1.0, 1.25, 1.25, 1.35],
            wspace=0.22,
        )

        # Precision chassis hairline dividers
        self.fig.add_artist(
            plt.Line2D([0.035, 0.965], [0.930, 0.930], color="#242c38", linewidth=1.0, transform=self.fig.transFigure)
        )
        self.fig.add_artist(
            plt.Line2D([0.035, 0.965], [0.430, 0.430], color="#242c38", linewidth=1.0, transform=self.fig.transFigure)
        )

    def _create_dark_slider(
        self,
        cell,
        title: str,
        valmin: float,
        valmax: float,
        valinit: float,
        valstep: float,
        valfmt: str,
        accent_color: str,
    ):
        """Construct a high-density, collision-free dark scientific slider."""
        gs_item = cell.subgridspec(2, 1, height_ratios=[0.45, 0.55], hspace=0.1)

        ax_lbl = self.fig.add_subplot(gs_item[0])
        ax_lbl.axis("off")
        ax_lbl.text(0.0, 0.1, title, fontsize=8.2, fontweight="bold", color="#8b949e", ha="left", va="bottom", family="monospace")
        val_txt = ax_lbl.text(1.0, 0.1, valfmt % valinit, fontsize=8.5, fontweight="bold", color=accent_color, ha="right", va="bottom", family="monospace")

        ax_track = self.fig.add_subplot(gs_item[1], facecolor="#161c24")
        for spine in ax_track.spines.values():
            spine.set_color("#242c38")
            spine.set_linewidth(0.6)

        slider = Slider(
            ax_track,
            "",
            valmin,
            valmax,
            valinit=valinit,
            valstep=valstep,
            color=accent_color,
            track_color="#161c24",
            initcolor="none",
        )
        slider.valtext.set_visible(False)

        def on_change(val):
            val_txt.set_text(valfmt % val)
            self.update(None)

        slider.on_changed(on_change)
        return slider, ax_lbl, ax_track, val_txt

    def build_controls(self) -> None:
        """Construct the top preset bar, 3 instrument control decks, and real-time telemetry terminal."""
        # ─────────────────────────────────────────────────────────────
        # Top Header Bar Content
        # ─────────────────────────────────────────────────────────────
        self.ax_head_brand.text(0.0, 0.85, "GRATINGLAB // OPTICAL SPECTROMETER", fontsize=10.5, fontweight="bold", color="#f0f6fc", family="sans-serif")
        self.ax_head_brand.text(0.0, 0.15, "SYS-ID: GL-650X-PRO  |  PRECISION OPTICAL METROLOGY", fontsize=7.6, color="#8b949e", family="monospace")

        self.lbl_head_status = self.ax_head_status.text(0.5, 0.85, "● SYSTEM ONLINE", fontsize=8.8, fontweight="bold", color="#10b981", family="monospace", ha="center")
        self.lbl_head_telemetry = self.ax_head_status.text(0.5, 0.15, "LIVE // λ = 650 nm | d = 1.60 µm", fontsize=7.6, color="#64748b", family="monospace", ha="center")

        # Top Instrument Action Buttons
        self.btn_preset1 = Button(self.fig.add_subplot(self.gs_hbtns[0]), "Single Slit", color="#161c24", hovercolor="#242c38")
        self.btn_preset1.label.set_fontsize(7.5)
        self.btn_preset1.label.set_color("#e6edf3")
        self.btn_preset1.on_clicked(lambda _e: self.apply_preset("Single slit"))

        self.btn_preset2 = Button(self.fig.add_subplot(self.gs_hbtns[1]), "Grating (CD)", color="#161c24", hovercolor="#242c38")
        self.btn_preset2.label.set_fontsize(7.5)
        self.btn_preset2.label.set_color("#e6edf3")
        self.btn_preset2.on_clicked(lambda _e: self.apply_preset("Diffraction grating"))

        self.btn_preset3 = Button(self.fig.add_subplot(self.gs_hbtns[2]), "Blazed Spec", color="#161c24", hovercolor="#242c38")
        self.btn_preset3.label.set_fontsize(7.5)
        self.btn_preset3.label.set_color("#e6edf3")
        self.btn_preset3.on_clicked(lambda _e: self.apply_preset("Blazed spectrometer"))

        self.btn_talbot = Button(self.fig.add_subplot(self.gs_hbtns[3]), "Talbot 2D", color="#0c2b2e", hovercolor="#115e59")
        self.btn_talbot.label.set_fontsize(7.5)
        self.btn_talbot.label.set_fontweight("bold")
        self.btn_talbot.label.set_color("#2dd4bf")
        self.btn_talbot.on_clicked(self.open_talbot_window)

        self.btn_reset = Button(self.fig.add_subplot(self.gs_hbtns[4]), "Reset", color="#2c1517", hovercolor="#4c1d24")
        self.btn_reset.label.set_fontsize(7.5)
        self.btn_reset.label.set_fontweight("bold")
        self.btn_reset.label.set_color("#fca5a5")
        self.btn_reset.on_clicked(self.reset)

        # ─────────────────────────────────────────────────────────────
        # Section 01: Grating Profile & Optical Mode (Theme: Sky / Teal)
        # ─────────────────────────────────────────────────────────────
        gs_c0 = self.gs_controls[0].subgridspec(4, 1, height_ratios=[0.4, 1.8, 0.35, 1.0], hspace=0.15)

        ax_h0 = self.fig.add_subplot(gs_c0[0])
        ax_h0.axis("off")
        ax_h0.text(0.0, 0.85, "SECTION 01 // GRATING PROFILE", fontsize=8.5, fontweight="bold", color="#38bdf8", family="monospace")
        ax_h0.text(0.0, 0.1, "Aperture modulation function", fontsize=7.5, color="#64748b")

        ax_radio = self.fig.add_subplot(gs_c0[1], facecolor="#11161d")
        for spine in ax_radio.spines.values():
            spine.set_color("#242c38")
            spine.set_linewidth(0.8)

        self.radio_type = RadioButtons(
            ax_radio,
            ("Binary Amplitude", "Sinusoidal Phase", "Blazed Phase", "2D Orthogonal"),
            active=["Binary Amplitude", "Sinusoidal Phase", "Blazed Phase", "2D Orthogonal"].index(self.grating_type),
            activecolor="#14b8a6",
        )
        for label in self.radio_type.labels:
            label.set_color("#e6edf3")
            label.set_fontsize(8)
        self.radio_type.on_clicked(self.on_type)

        ax_h0_sub = self.fig.add_subplot(gs_c0[2])
        ax_h0_sub.axis("off")
        ax_h0_sub.text(0.0, 0.1, "VIEW MODES:", fontsize=8, fontweight="bold", color="#8b949e", family="monospace")

        ax_checks = self.fig.add_subplot(gs_c0[3], facecolor="#11161d")
        for spine in ax_checks.spines.values():
            spine.set_color("#242c38")
            spine.set_linewidth(0.8)

        self.check_toggles = CheckButtons(
            ax_checks,
            ("Log Scale", "Compare Analytical"),
            (self.log_scale, self.compare_analytical),
        )
        for label in self.check_toggles.labels:
            label.set_color("#e6edf3")
            label.set_fontsize(8)
        self.check_toggles.on_clicked(self.on_toggle)

        # ─────────────────────────────────────────────────────────────
        # Section 02: Laser Emitter (Theme: Laboratory Orange #f97316)
        # ─────────────────────────────────────────────────────────────
        gs_c1 = self.gs_controls[1].subgridspec(5, 1, height_ratios=[0.4, 0.85, 0.85, 0.85, 0.85], hspace=0.35)

        ax_h1 = self.fig.add_subplot(gs_c1[0])
        ax_h1.axis("off")
        ax_h1.text(0.0, 0.85, "SECTION 02 // LASER EMITTER", fontsize=8.5, fontweight="bold", color="#f97316", family="monospace")
        ax_h1.text(0.0, 0.1, "Coherent laser excitation", fontsize=7.5, color="#64748b")

        self.s_wl, _, _, self.txt_wl = self._create_dark_slider(
            gs_c1[1], "WAVELENGTH λ", 400.0, 700.0, self.wavelength * 1e9, 1.0, "%0.0f nm", "#f97316"
        )
        self.s_theta, _, _, self.txt_theta = self._create_dark_slider(
            gs_c1[2], "INCIDENCE ANGLE θᵢ", -30.0, 30.0, math.degrees(self.theta_i), 0.5, "%0.1f°", "#f97316"
        )
        self.s_waist, _, _, self.txt_waist = self._create_dark_slider(
            gs_c1[3], "BEAM WAIST w₀", 5.0, 100.0, self.beam_waist * 1e6, 1.0, "%0.1f µm", "#f97316"
        )
        self.s_dist, _, _, self.txt_dist = self._create_dark_slider(
            gs_c1[4], "SCREEN DIST L", 0.1, 2.0, self.screen_dist, 0.05, "%0.2f m", "#f97316"
        )

        # ─────────────────────────────────────────────────────────────
        # Section 03: Microstructure Geometry (Theme: Laboratory Teal #14b8a6)
        # ─────────────────────────────────────────────────────────────
        gs_c2 = self.gs_controls[2].subgridspec(5, 1, height_ratios=[0.4, 0.85, 0.85, 0.85, 0.85], hspace=0.35)

        ax_h2 = self.fig.add_subplot(gs_c2[0])
        ax_h2.axis("off")
        ax_h2.text(0.0, 0.85, "SECTION 03 // GRATING STRUCTURE", fontsize=8.5, fontweight="bold", color="#14b8a6", family="monospace")
        ax_h2.text(0.0, 0.1, "Diffractive microstructure pitch", fontsize=7.5, color="#64748b")

        self.s_period, _, _, self.txt_period = self._create_dark_slider(
            gs_c2[1], "GRATING PITCH d", 0.5, 20.0, self.period * 1e6, 0.05, "%0.2f µm", "#14b8a6"
        )
        self.s_n, _, _, self.txt_n = self._create_dark_slider(
            gs_c2[2], "RULINGS COUNT N", 2, 50, self.n_slits, 1, "%d lines", "#14b8a6"
        )

        # Progressive disclosure slot (row 3): Duty vs Phase Depth
        self.s_duty, self.ax_duty_lbl, self.ax_duty_track, self.txt_duty = self._create_dark_slider(
            gs_c2[3], "DUTY / SLIT FRAC", 0.05, 0.95, self.duty, 0.01, "%0.2f", "#14b8a6"
        )
        self.s_phase, self.ax_phase_lbl, self.ax_phase_track, self.txt_phase = self._create_dark_slider(
            gs_c2[3], "PHASE DEPTH φ₀", 0.1, 4.0 * math.pi, self.phase_depth, 0.1, "%0.2f rad", "#14b8a6"
        )

        ax_g_hint = self.fig.add_subplot(gs_c2[4])
        ax_g_hint.axis("off")
        self.lbl_grating_freq = ax_g_hint.text(0.0, 0.65, "", fontsize=7.8, color="#38bdf8", family="monospace")
        self.lbl_grating_hint = ax_g_hint.text(0.0, 0.15, "", fontsize=7.5, color="#64748b")

        # ─────────────────────────────────────────────────────────────
        # Section 04: Bloomberg-Grade Telemetry Terminal (Theme: Cyan #38bdf8)
        # ─────────────────────────────────────────────────────────────
        gs_c3 = self.gs_controls[3].subgridspec(2, 1, height_ratios=[0.4, 3.4], hspace=0.08)

        ax_h3 = self.fig.add_subplot(gs_c3[0])
        ax_h3.axis("off")
        ax_h3.text(0.0, 0.85, "SECTION 04 // OPTICAL TELEMETRY", fontsize=8.5, fontweight="bold", color="#38bdf8", family="monospace")
        ax_h3.text(0.0, 0.1, "Real-time Fraunhofer order analysis", fontsize=7.5, color="#64748b")

        self.ax_term = self.fig.add_subplot(gs_c3[1], facecolor="#090d12")
        for spine in self.ax_term.spines.values():
            spine.set_color("#242c38")
            spine.set_linewidth(0.8)
        self.ax_term.set_xticks([])
        self.ax_term.set_yticks([])

        self.term_text = self.ax_term.text(
            0.04, 0.94, "",
            va="top", ha="left", fontsize=7.3, family="monospace", color="#7dd3fc",
            linespacing=1.28
        )

    def on_type(self, label: str) -> None:
        """Handle grating profile switching and progressive disclosure."""
        self.grating_type = label
        is_phase = label in ("Sinusoidal Phase", "Blazed Phase")

        if is_phase:
            self.ax_duty_lbl.set_visible(False)
            self.ax_duty_track.set_visible(False)
            self.ax_phase_lbl.set_visible(True)
            self.ax_phase_track.set_visible(True)
        else:
            self.ax_phase_lbl.set_visible(False)
            self.ax_phase_track.set_visible(False)
            self.ax_duty_lbl.set_visible(True)
            self.ax_duty_track.set_visible(True)

        hints = {
            "Binary Amplitude": "Profile: Binary chrome-on-glass slits",
            "Sinusoidal Phase": "Profile: Sinusoidal phase modulation",
            "Blazed Phase": "Profile: Asymmetric sawtooth blazed grooves",
            "2D Orthogonal": "Profile: 2D orthogonal grid mesh",
        }
        self.lbl_grating_hint.set_text(hints.get(label, ""))
        self.update(None)

    def on_toggle(self, _label: str) -> None:
        """Handle view check box toggles."""
        status = self.check_toggles.get_status()
        self.log_scale = status[0]
        self.compare_analytical = status[1]
        self.update(None)

    def reset(self, _event=None) -> None:
        """Restore all parameters to defaults."""
        d = self.default_params
        self.apply_values(
            grating_type=d["grating_type"],
            wl=d["wavelength_nm"],
            period=d["period_um"],
            n_slits=d["n_slits"],
            duty=d["duty"],
            phase=d["phase_depth"],
            theta=d["theta_i_deg"],
            waist=d["beam_waist_um"],
            dist=d["screen_dist_m"],
            log=d["log_scale"],
            comp=d["compare_analytical"],
        )

    def apply_preset(self, name: str) -> None:
        """Apply pre-configured optical presets."""
        if name == "Single slit":
            self.apply_values(
                grating_type="Binary Amplitude",
                wl=650.0,
                period=4.0,
                n_slits=1,
                duty=0.5,
                theta=0.0,
                waist=50.0,
                dist=0.5,
            )
        elif name == "Diffraction grating":
            self.apply_values(
                grating_type="Binary Amplitude",
                wl=650.0,
                period=1.6,
                n_slits=20,
                duty=0.5,
                theta=0.0,
                waist=40.0,
                dist=0.5,
            )
        elif name == "Blazed spectrometer":
            self.apply_values(
                grating_type="Blazed Phase",
                wl=650.0,
                period=1.6,
                n_slits=30,
                phase=2.0 * math.pi,
                theta=0.0,
                waist=40.0,
                dist=0.5,
            )

    def apply_values(
        self,
        grating_type: str | None = None,
        wl: float | None = None,
        period: float | None = None,
        n_slits: int | None = None,
        duty: float | None = None,
        phase: float | None = None,
        theta: float | None = None,
        waist: float | None = None,
        dist: float | None = None,
        log: bool | None = None,
        comp: bool | None = None,
    ) -> None:
        """Programmatically update widgets and state without redundant event floods."""
        sliders = [self.s_wl, self.s_theta, self.s_waist, self.s_dist, self.s_period, self.s_n, self.s_duty, self.s_phase]
        for s in sliders:
            s.eventson = False

        if wl is not None:
            self.s_wl.set_val(wl)
            self.txt_wl.set_text("%0.0f nm" % wl)
        if theta is not None:
            self.s_theta.set_val(theta)
            self.txt_theta.set_text("%0.1f°" % theta)
        if waist is not None:
            self.s_waist.set_val(waist)
            self.txt_waist.set_text("%0.1f µm" % waist)
        if dist is not None:
            self.s_dist.set_val(dist)
            self.txt_dist.set_text("%0.2f m" % dist)
        if period is not None:
            self.s_period.set_val(period)
            self.txt_period.set_text("%0.2f µm" % period)
        if n_slits is not None:
            self.s_n.set_val(n_slits)
            self.txt_n.set_text("%d lines" % n_slits)
        if duty is not None:
            self.s_duty.set_val(duty)
            self.txt_duty.set_text("%0.2f" % duty)
        if phase is not None:
            self.s_phase.set_val(phase)
            self.txt_phase.set_text("%0.2f rad" % phase)

        for s in sliders:
            s.eventson = True

        if log is not None or comp is not None:
            status = self.check_toggles.get_status()
            if log is not None and status[0] != log:
                self.check_toggles.set_active(0)
            if comp is not None and status[1] != comp:
                self.check_toggles.set_active(1)

        if grating_type is not None and grating_type != self.grating_type:
            types = ["Binary Amplitude", "Sinusoidal Phase", "Blazed Phase", "2D Orthogonal"]
            if grating_type in types:
                self.radio_type.set_active(types.index(grating_type))
                return

        self.on_type(self.grating_type)

    def update(self, _val=None) -> None:
        """Perform exact wave-optics calculation, update research-grade plots, and telemetry terminal."""
        self.wavelength = self.s_wl.val * 1e-9
        self.period = self.s_period.val * 1e-6
        self.duty = self.s_duty.val
        self.phase_depth = self.s_phase.val
        self.n_slits = int(self.s_n.val)
        self.theta_i = math.radians(self.s_theta.val)
        self.screen_dist = self.s_dist.val
        self.beam_waist = self.s_waist.val * 1e-6

        # Update spatial frequency label
        spatial_freq = 1.0 / (self.period * 1e3) if self.period > 0 else 0.0
        self.lbl_grating_freq.set_text(f"SPATIAL FREQUENCY: {spatial_freq:.1f} lines/mm")

        # Update top status telemetry
        self.lbl_head_telemetry.set_text(f"LIVE // λ = {self.s_wl.val:.0f} nm | d = {self.period*1e6:.2f} µm")

        display_periods = max(12, int(self.n_slits * 1.5))
        window_width = display_periods * self.period
        grid = gratings.Grid1D(width=window_width, num_samples=self.num_samples)

        a = self.duty * self.period
        if self.grating_type == "Binary Amplitude":
            t_raw = gratings.binary_amplitude(grid.x, d=self.period, a=a)
        elif self.grating_type == "Sinusoidal Phase":
            t_raw = gratings.sinusoidal_phase(grid.x, d=self.period, phi0=self.phase_depth)
        elif self.grating_type == "Blazed Phase":
            t_raw = gratings.blazed(grid.x, d=self.period, phase_depth=self.phase_depth)
        else:
            t_raw = gratings.binary_amplitude(grid.x, d=self.period, a=a)

        aperture_mask = gratings.finite_aperture(grid.x, n_lines=self.n_slits, d=self.period)
        incident_beam = propagation.gaussian_beam(grid.x, w0=self.beam_waist)
        t_illuminated = t_raw * aperture_mask * incident_beam

        # ─── 1. Channel 1: Aperture Plane Transmittance & Phase ───
        self.ax_profile.clear()
        self.ax_profile.set_facecolor("#11161d")
        x_um = grid.x * 1e6
        crop_mask = np.abs(grid.x) <= (4.0 * self.period)
        d_um = self.period * 1e6

        # Shaded magnitude area fill & sharp outline
        self.ax_profile.fill_between(x_um[crop_mask], 0, np.abs(t_raw)[crop_mask], color="#14b8a6", alpha=0.18, label="|t(x)| Magnitude")
        self.ax_profile.plot(x_um[crop_mask], np.abs(t_raw)[crop_mask], color="#2dd4bf", lw=1.8)
        self.ax_profile.plot(x_um[crop_mask], np.angle(t_raw)[crop_mask] / math.pi, color="#f97316", ls="--", lw=1.4, label="Phase φ(x) / π")
        self.ax_profile.plot(x_um[crop_mask], np.abs(incident_beam)[crop_mask], color="#94a3b8", ls=":", lw=1.2, alpha=0.75, label="Beam Envelope w₀")

        # Period dimension annotation bracket
        if 0.5 <= d_um <= (np.sum(crop_mask) * (grid.dx * 1e6)):
            self.ax_profile.annotate(
                "", xy=(0.0, 1.08), xytext=(d_um, 1.08),
                arrowprops=dict(arrowstyle="<->", color="#38bdf8", lw=1.0)
            )
            self.ax_profile.text(
                d_um / 2.0, 1.12, f"d = {d_um:.2f} µm",
                ha="center", va="bottom", fontsize=7.5, color="#38bdf8", family="monospace", fontweight="bold"
            )

        self.ax_profile.set_title(f"CH1 // APERTURE TRANSMITTANCE & PHASE MODULATION: {self.grating_type.upper()}", fontsize=9, fontweight="bold", color="#e6edf3", pad=10, loc="left")
        self.ax_profile.set_xlabel("Transverse Coordinate x [µm]", fontsize=8, color="#8b949e", labelpad=2)
        self.ax_profile.set_ylabel("Field Amplitude / Phase (rad/π)", fontsize=8, color="#8b949e")
        self.ax_profile.set_ylim(-1.15, 1.28)
        self.ax_profile.grid(True, linestyle=":", color="#212936", alpha=0.8)
        leg1 = self.ax_profile.legend(loc="upper right", fontsize=7.2, facecolor="#161c24", edgecolor="#2c3543")
        for text in leg1.get_texts():
            text.set_color("#e6edf3")

        # ─── 2. Channel 2: Far-Field Fraunhofer Diffraction Spectrum ───
        sin_theta, intensity, theta_deg = propagation.far_field_fft(
            t_illuminated,
            dx=grid.dx,
            wavelength=self.wavelength,
            pad_factor=2,
        )
        theta_rad_dense = np.radians(theta_deg)

        self.ax_fft.clear()
        self.ax_fft.set_facecolor("#11161d")
        y_data = np.maximum(intensity, 1e-4) if self.log_scale else intensity

        # Spectrum area fill & primary trace
        self.ax_fft.fill_between(theta_deg, 1e-4 if self.log_scale else 0, y_data, color="#10b981", alpha=0.12)
        self.ax_fft.plot(theta_deg, y_data, label="Numerical FFT", color="#10b981", lw=1.6)

        # Analytical multi-slit model
        if self.compare_analytical and self.grating_type == "Binary Amplitude":
            i_analytical = analytical.n_slit_intensity(
                theta=theta_rad_dense,
                a=a,
                d=self.period,
                N=self.n_slits,
                wavelength=self.wavelength,
                theta_i=self.theta_i,
            )
            y_analytical = np.maximum(i_analytical, 1e-4) if self.log_scale else i_analytical
            self.ax_fft.plot(theta_deg, y_analytical, label="Analytical N-Slit", color="#f59e0b", ls=":", lw=1.3)

            # Single-slit diffraction envelope sinc²(β)
            beta_norm = (a / self.wavelength) * np.sin(theta_rad_dense)
            sinc_sq = np.sinc(beta_norm)**2
            y_env = np.maximum(sinc_sq, 1e-4) if self.log_scale else sinc_sq
            self.ax_fft.plot(theta_deg, y_env, label="Slit Envelope sinc²(β)", color="#64748b", ls="-.", lw=1.1, alpha=0.8)

        # Propagating diffraction orders
        orders, evanescent = analytical.order_angles(
            d=self.period,
            wavelength=self.wavelength,
            theta_i=self.theta_i,
            m_max=4,
        )
        for m, ang_rad in orders.items():
            ang_d = math.degrees(ang_rad)
            if -48.0 <= ang_d <= 48.0:
                self.ax_fft.axvline(ang_d, color="#334155", linestyle="--", alpha=0.6, lw=0.9)
                self.ax_fft.text(
                    ang_d,
                    1.01,
                    f"m={m:+d}" if m != 0 else "m=0",
                    transform=self.ax_fft.get_xaxis_transform(),
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    color="#38bdf8",
                    fontweight="bold",
                    family="monospace",
                )
                # Diffraction peak tip marker
                peak_idx = np.argmin(np.abs(theta_deg - ang_d))
                peak_val = y_data[peak_idx]
                if peak_val > (1e-3 if self.log_scale else 0.05):
                    marker_col = "#f97316" if (self.grating_type == "Blazed Phase" and m == 1) else "#38bdf8"
                    self.ax_fft.scatter(ang_d, peak_val, color=marker_col, s=18, zorder=4, edgecolor="#0b0f14", linewidth=0.5)

        # Suppressed missing orders markers (e.g. m = ±2)
        if self.grating_type == "Binary Amplitude":
            missing_list = analytical.missing_orders(self.period, a)
            for m_miss in missing_list:
                if m_miss in orders:
                    ang_miss_d = math.degrees(orders[m_miss])
                    if -48.0 <= ang_miss_d <= 48.0:
                        self.ax_fft.scatter(ang_miss_d, 1e-4 if self.log_scale else 0.0, marker="x", color="#ef4444", s=22, zorder=5, lw=1.2)
                        self.ax_fft.text(
                            ang_miss_d, 2e-4 if self.log_scale else 0.04, f"m={m_miss:+} (0)",
                            ha="center", va="bottom", fontsize=6.8, color="#ef4444", family="monospace"
                        )

        self.ax_fft.set_title("CH2 // FAR-FIELD FRAUNHOFER DIFFRACTION SPECTRUM", fontsize=9, fontweight="bold", color="#e6edf3", pad=10, loc="left")
        self.ax_fft.set_xlabel("Diffraction Angle θ [deg]", fontsize=8, color="#8b949e", labelpad=2)
        self.ax_fft.set_ylabel("Relative Log Irradiance log₁₀(I / I₀)" if self.log_scale else "Normalized Spectral Irradiance I(θ)", fontsize=8, color="#8b949e")
        self.ax_fft.set_xlim(-50.0, 50.0)
        if self.log_scale:
            self.ax_fft.set_yscale("log")
            self.ax_fft.set_ylim(1e-4, 1.25)
        else:
            self.ax_fft.set_ylim(-0.02, 1.08)
        self.ax_fft.grid(True, linestyle=":", color="#212936", alpha=0.8)
        leg2 = self.ax_fft.legend(loc="upper right", fontsize=7.2, facecolor="#161c24", edgecolor="#2c3543")
        for text in leg2.get_texts():
            text.set_color("#e6edf3")

        # ─── 3. Detector Array: Simulated Screen Pattern ───
        self.ax_screen.clear()
        screen_y_cm = propagation.sin_theta_to_screen_position(np.radians(theta_deg), self.screen_dist) * 100.0
        y_screen_linear = np.linspace(-30.0, 30.0, 600)
        i_screen = np.interp(y_screen_linear, screen_y_cm, intensity, left=0.0, right=0.0)

        # Realistic Gaussian vertical cross-section for laser spots
        y_vert = np.linspace(-1.0, 1.0, 35)[:, np.newaxis]
        strip_2d = np.exp(-y_vert**2 / 0.35) * i_screen[np.newaxis, :]

        laser_cmap = get_laser_colormap(self.s_wl.val)
        self.ax_screen.imshow(
            strip_2d,
            extent=[-30.0, 30.0, -1.5, 1.5],
            cmap=laser_cmap,
            aspect="auto",
            origin="lower",
        )

        # Spot location pointer triangles
        for m, ang_rad in orders.items():
            y_m = self.screen_dist * math.tan(ang_rad) * 100.0
            if -28.0 <= y_m <= 28.0:
                self.ax_screen.scatter(y_m, 1.35, marker="v", color="#38bdf8", s=16, zorder=5)
                self.ax_screen.text(
                    y_m, 1.48, f"m={m:+d}" if m != 0 else "m=0",
                    ha="center", va="bottom", fontsize=6.8, color="#38bdf8", family="monospace"
                )

        self.ax_screen.set_title(
            f"DETECTOR ARRAY // SIMULATED SCREEN PATTERN (L = {self.screen_dist:.2f} m  |  λ = {int(self.s_wl.val)} nm  |  y = L·tan θ)",
            fontsize=8.5,
            fontweight="bold",
            color="#e6edf3",
            pad=8,
            loc="left",
        )
        self.ax_screen.set_xlabel("Transverse Screen Displacement y [cm] (Metric Scale)", fontsize=8, color="#8b949e", labelpad=2)
        self.ax_screen.set_ylim(-1.5, 2.0)
        self.ax_screen.set_yticks([])

        # ─── 4. Bloomberg Telemetry Terminal Feed ───
        z_t_um = (2.0 * self.period**2 / self.wavelength) * 1e6
        telemetry_lines = [
            f"SYS: λ={self.wavelength*1e9:.0f}nm  d={self.period*1e6:.2f}µm  N={self.n_slits}  L={self.screen_dist:.2f}m",
            f"FREQ: {spatial_freq:.0f} lines/mm  |  z_T = {z_t_um:.2f} µm",
            "─" * 38,
            "ORD   DIFF-ANGLE    DET-OFFSET   DISPERSION",
        ]
        for m in sorted(orders.keys()):
            ang_deg = math.degrees(orders[m])
            y_cm = self.screen_dist * math.tan(orders[m]) * 100.0
            cos_th = math.cos(orders[m])
            disp = (abs(m) / (self.period * cos_th)) * (180.0 / math.pi) * 1e-9 if cos_th > 1e-4 else 0.0
            blaze_flag = "  ◀ BLAZE" if (self.grating_type == "Blazed Phase" and m == 1) else ""
            telemetry_lines.append(f"m={m:+2d}   {ang_deg:+7.2f}°    {y_cm:+7.1f} cm    {disp:0.3f}°/nm{blaze_flag}")

        telemetry_lines.append("─" * 38)
        if self.grating_type == "Binary Amplitude":
            missing = analytical.missing_orders(self.period, a)
            if missing:
                missing_str = ", ".join(f"±{x}" for x in missing[:4])
                telemetry_lines.append(f"MISSING ORDERS : m = {missing_str} ({int(self.duty*100)}% duty)")
            else:
                telemetry_lines.append("MISSING ORDERS : None detected")
        elif self.grating_type == "Blazed Phase":
            telemetry_lines.append("PRIMARY BLAZE  : m = +1 (Efficiency ~95%)")
        elif self.grating_type == "Sinusoidal Phase":
            telemetry_lines.append(f"PHASE DEPTH    : φ₀ = {self.phase_depth:.2f} rad (Bessel J_m)")
        else:
            telemetry_lines.append("2D PROFILE     : 2-axis orthogonal grid")

        if evanescent:
            telemetry_lines.append(f"EVANESCENT     : |m| ≥ {min(abs(e) for e in evanescent)} (sin θ > 1)")
        else:
            telemetry_lines.append("EVANESCENT     : All modeled orders propagate")

        self.term_text.set_text("\n".join(telemetry_lines))
        self.fig.canvas.draw_idle()

    def open_talbot_window(self, _event=None) -> None:
        """Launch the near-field Talbot carpet simulation in a dark research-grade window."""
        fig_t = plt.figure(figsize=(10, 6.5), dpi=100, facecolor="#0b0f14")
        if hasattr(fig_t.canvas, "manager") and fig_t.canvas.manager is not None:
            fig_t.canvas.manager.set_window_title("GratingLab // Near-Field Talbot Carpet Analyzer")
        ax_t = fig_t.add_subplot(111, facecolor="#11161d")

        z_t = analytical.talbot_distance(self.period, self.wavelength)
        grid = gratings.Grid1D(width=20.0 * self.period, num_samples=2048)
        t_grating = gratings.binary_amplitude(grid.x, d=self.period, a=self.duty * self.period)

        x, z_arr, carpet = propagation.talbot_carpet(
            field=t_grating,
            dx=grid.dx,
            wavelength=self.wavelength,
            d=self.period,
            n_z_steps=250,
            z_max=2.0 * z_t,
        )

        crop = np.abs(x) <= (3.0 * self.period)
        x_crop_um = x[crop] * 1e6
        z_max_um = z_arr[-1] * 1e6
        z_t_um = z_t * 1e6

        im = ax_t.imshow(
            carpet[:, crop],
            extent=[x_crop_um[0], x_crop_um[-1], 0.0, z_max_um],
            aspect="auto",
            origin="lower",
            cmap="inferno",
        )

        # Fractional Talbot sub-plane markers
        ax_t.axhline(z_t_um, color="#2dd4bf", ls="-", lw=1.4, label=f"z = z_T (Full Self-Image, {z_t_um:.2f} µm)")
        ax_t.axhline(z_t_um * 0.5, color="#f59e0b", ls="--", lw=1.2, label=f"z = z_T/2 (Half-Talbot Shift, {z_t_um*0.5:.2f} µm)")
        ax_t.axhline(z_t_um * 0.25, color="#f97316", ls=":", lw=1.2, label=f"z = z_T/4 (Frequency Doubling, {z_t_um*0.25:.2f} µm)")

        # Secondary y-axis for normalized propagation distance z / z_T
        ax_norm = ax_t.twinx()
        ax_norm.set_ylim(0.0, 2.0)
        ax_norm.set_ylabel("Normalized Distance z / z_T", fontsize=8.5, color="#38bdf8")
        ax_norm.tick_params(colors="#38bdf8", labelsize=8)
        for spine in ax_norm.spines.values():
            spine.set_color("#242c38")

        ax_t.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax_t.spines.values():
            spine.set_color("#242c38")
            spine.set_linewidth(0.8)

        ax_t.set_title(
            f"NEAR-FIELD ASM WAVEFIELD PROPAGATION // TALBOT SELF-IMAGING PHENOMENON\n"
            f"Parameters: Pitch d = {self.period*1e6:.2f} µm  |  Wavelength λ = {self.wavelength*1e9:.0f} nm  |  Talbot Distance z_T = {z_t_um:.2f} µm",
            fontsize=9.2,
            fontweight="bold",
            color="#e6edf3",
            pad=12,
            loc="left",
        )
        ax_t.set_xlabel("Transverse Coordinate x [µm]", fontsize=8.5, color="#8b949e")
        ax_t.set_ylabel("Propagation Distance z [µm]", fontsize=8.5, color="#8b949e")

        leg = ax_t.legend(loc="upper right", fontsize=7.8, facecolor="#161c24", edgecolor="#2c3543")
        for t in leg.get_texts():
            t.set_color("#e6edf3")

        cbar = fig_t.colorbar(im, ax=ax_norm, pad=0.08)
        cbar.ax.tick_params(colors="#8b949e", labelsize=8)
        cbar.set_label("Normalized Wavefield Irradiance |U(x, z)|²", fontsize=8.5, color="#8b949e")
        cbar.outline.set_edgecolor("#242c38")

        plt.tight_layout()
        plt.show()

    # Backwards compatibility alias
    _open_talbot_window = open_talbot_window


def launch_interactive_app(
    grating_type: str = "Binary Amplitude",
    wavelength_nm: float = 650.0,
    period_um: float = 1.6,
    n_slits: int = 10,
    screen_dist_m: float = 0.5,
) -> None:
    current_backend = matplotlib.get_backend().lower()
    if current_backend in ("agg", "template", "pdf", "ps", "svg"):
        print("\n" + "!" * 72)
        print(f"  NOTICE: Matplotlib is using non-interactive backend '{current_backend}'.")
        print("  To open a desktop window, install an interactive backend such as PyQt6:")
        print("      pip install PyQt6")
        print("  or ensure Tkinter is installed in your Python environment.")
        print("!" * 72 + "\n")

    app = GratingLabApp(
        grating_type=grating_type,
        wavelength_nm=wavelength_nm,
        period_um=period_um,
        n_slits=n_slits,
        screen_dist_m=screen_dist_m,
    )
    plt.show()
