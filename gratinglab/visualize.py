"""
gratinglab.visualize
====================
Interactive Matplotlib dashboard for optical grating diffraction and characterization.
Redesigned with GridSpec layout, 3 vertical zones, 4 structured control groups,
progressive disclosure, and collision-free slider typography.
"""

from __future__ import annotations
import math
import os
import sys

# Ensure an interactive backend is used for desktop GUI rendering
import matplotlib
if matplotlib.get_backend().lower() in ("agg", "template", "pdf", "ps", "svg"):
    for _backend in ["TkAgg", "QtAgg", "Qt5Agg", "WXAgg"]:
        try:
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
        """Create pure GridSpec architecture in three vertical zones without manual coordinates."""
        plt.style.use("fast")
        self.fig = plt.figure(figsize=(16, 9.5), dpi=100)
        self.fig.canvas.manager.set_window_title("GratingLab: Optical Diffraction Studio")

        # 3 Vertical Zones:
        # Zone 1: Aperture profile and far-field diffraction plots
        # Zone 2: Simulated laser screen strip
        # Zone 3: 4-Column control dashboard
        self.gs_main = self.fig.add_gridspec(
            3, 1,
            height_ratios=[3.6, 0.8, 3.8],
            hspace=0.34,
            left=0.045, right=0.965, top=0.95, bottom=0.045,
        )

        # Zone 1: Top Plots
        gs_top = self.gs_main[0].subgridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.18)
        self.ax_profile = self.fig.add_subplot(gs_top[0])
        self.ax_fft = self.fig.add_subplot(gs_top[1])

        # Zone 2: Screen Strip
        self.ax_screen = self.fig.add_subplot(self.gs_main[1])

        # Zone 3: 4 Control Columns
        self.gs_controls = self.gs_main[2].subgridspec(
            1, 4,
            width_ratios=[1.0, 1.25, 1.25, 1.2],
            wspace=0.22,
        )

    def _create_slider(
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
        """Build an anti-collision slider with title on left and value on right above the slim track."""
        gs_item = cell.subgridspec(2, 1, height_ratios=[0.45, 0.55], hspace=0.1)

        ax_lbl = self.fig.add_subplot(gs_item[0])
        ax_lbl.axis("off")
        ax_lbl.text(0.0, 0.1, title, fontsize=9, fontweight="bold", color="#1e293b", ha="left", va="bottom")
        val_txt = ax_lbl.text(1.0, 0.1, valfmt % valinit, fontsize=9, fontweight="bold", color="#1e293b", ha="right", va="bottom")

        ax_track = self.fig.add_subplot(gs_item[1])
        slider = Slider(
            ax_track,
            "",
            valmin,
            valmax,
            valinit=valinit,
            valstep=valstep,
            color=accent_color,
            track_color="#e2e8f0",
            initcolor="none",
        )
        slider.valtext.set_visible(False)

        def on_change(val):
            val_txt.set_text(valfmt % val)
            self.update(None)

        slider.on_changed(on_change)
        return slider, ax_lbl, ax_track, val_txt

    def build_controls(self) -> None:
        """Construct the 4 control groups with consistent color coding and clear labels."""
        # ─────────────────────────────────────────────────────────────
        # Column 1: Grating Type & View Options (Theme: Navy #1e3a8a)
        # ─────────────────────────────────────────────────────────────
        gs_col0 = self.gs_controls[0].subgridspec(4, 1, height_ratios=[0.45, 1.8, 0.35, 1.0], hspace=0.15)

        ax_h0 = self.fig.add_subplot(gs_col0[0])
        ax_h0.axis("off")
        ax_h0.text(0.0, 0.9, "1. Grating Type", fontsize=10, fontweight="bold", color="#1e3a8a")
        ax_h0.text(0.0, 0.1, "Profile & View Options", fontsize=8, fontstyle="italic", color="#64748b")

        ax_radio = self.fig.add_subplot(gs_col0[1])
        self.radio_type = RadioButtons(
            ax_radio,
            ("Binary Amplitude", "Sinusoidal Phase", "Blazed Phase", "2D Orthogonal"),
            active=["Binary Amplitude", "Sinusoidal Phase", "Blazed Phase", "2D Orthogonal"].index(self.grating_type),
            activecolor="#2563eb",
        )
        self.radio_type.on_clicked(self.on_type)

        ax_h0_sub = self.fig.add_subplot(gs_col0[2])
        ax_h0_sub.axis("off")
        ax_h0_sub.text(0.0, 0.1, "View Options:", fontsize=8.5, fontweight="bold", color="#334155")

        ax_checks = self.fig.add_subplot(gs_col0[3])
        self.check_toggles = CheckButtons(
            ax_checks,
            ("Log Scale", "Compare Analytical"),
            (self.log_scale, self.compare_analytical),
        )
        self.check_toggles.on_clicked(self.on_toggle)

        # ─────────────────────────────────────────────────────────────
        # Column 2: Light Source (Theme: Orange #c2410c)
        # ─────────────────────────────────────────────────────────────
        gs_col1 = self.gs_controls[1].subgridspec(5, 1, height_ratios=[0.45, 0.85, 0.85, 0.85, 0.85], hspace=0.35)

        ax_h1 = self.fig.add_subplot(gs_col1[0])
        ax_h1.axis("off")
        ax_h1.text(0.0, 0.9, "2. Light Source", fontsize=10, fontweight="bold", color="#c2410c")
        ax_h1.text(0.0, 0.1, "Laser beam & screen setup", fontsize=8, fontstyle="italic", color="#64748b")

        self.s_wl, _, _, self.txt_wl = self._create_slider(
            gs_col1[1], "Wavelength [nm]", 400.0, 700.0, self.wavelength * 1e9, 1.0, "%0.0f", "#ea580c"
        )
        self.s_theta, _, _, self.txt_theta = self._create_slider(
            gs_col1[2], "Incidence θᵢ [°]", -30.0, 30.0, math.degrees(self.theta_i), 0.5, "%0.1f°", "#ea580c"
        )
        self.s_waist, _, _, self.txt_waist = self._create_slider(
            gs_col1[3], "Beam Waist [µm]", 5.0, 100.0, self.beam_waist * 1e6, 1.0, "%0.1f", "#ea580c"
        )
        self.s_dist, _, _, self.txt_dist = self._create_slider(
            gs_col1[4], "Screen Dist L [m]", 0.1, 2.0, self.screen_dist, 0.05, "%0.2f m", "#ea580c"
        )

        # ─────────────────────────────────────────────────────────────
        # Column 3: Grating Structure (Theme: Teal #0f766e)
        # ─────────────────────────────────────────────────────────────
        gs_col2 = self.gs_controls[2].subgridspec(5, 1, height_ratios=[0.45, 0.85, 0.85, 0.85, 0.85], hspace=0.35)

        ax_h2 = self.fig.add_subplot(gs_col2[0])
        ax_h2.axis("off")
        ax_h2.text(0.0, 0.9, "3. Grating Structure", fontsize=10, fontweight="bold", color="#0f766e")
        ax_h2.text(0.0, 0.1, "Microstructure geometry", fontsize=8, fontstyle="italic", color="#64748b")

        self.s_period, _, _, self.txt_period = self._create_slider(
            gs_col2[1], "Period d [µm]", 0.5, 20.0, self.period * 1e6, 0.05, "%0.2f", "#0d9488"
        )
        self.s_n, _, _, self.txt_n = self._create_slider(
            gs_col2[2], "Slits / Lines N", 2, 50, self.n_slits, 1, "%d", "#0d9488"
        )

        # Progressive disclosure slot (row 3): Duty vs Phase Depth
        self.s_duty, self.ax_duty_lbl, self.ax_duty_track, self.txt_duty = self._create_slider(
            gs_col2[3], "Duty / Slit frac", 0.05, 0.95, self.duty, 0.01, "%0.2f", "#0d9488"
        )
        self.s_phase, self.ax_phase_lbl, self.ax_phase_track, self.txt_phase = self._create_slider(
            gs_col2[3], "Phase Depth [rad]", 0.1, 4.0 * math.pi, self.phase_depth, 0.1, "%0.2f rad", "#0d9488"
        )

        ax_g_hint = self.fig.add_subplot(gs_col2[4])
        ax_g_hint.axis("off")
        self.lbl_grating_hint = ax_g_hint.text(0.0, 0.4, "", fontsize=8, color="#475569")

        # ─────────────────────────────────────────────────────────────
        # Column 4: Diagnostics & Presets (Theme: Charcoal #1e293b)
        # ─────────────────────────────────────────────────────────────
        gs_col3 = self.gs_controls[3].subgridspec(4, 1, height_ratios=[0.35, 1.85, 0.38, 0.72], hspace=0.16)

        ax_h3 = self.fig.add_subplot(gs_col3[0])
        ax_h3.axis("off")
        ax_h3.text(0.0, 0.9, "4. Diagnostics & Presets", fontsize=10, fontweight="bold", color="#1e293b")
        ax_h3.text(0.0, 0.1, "Readouts & quick setups", fontsize=8, fontstyle="italic", color="#64748b")

        ax_info = self.fig.add_subplot(gs_col3[1])
        ax_info.axis("off")
        self.info_text = ax_info.text(
            0.0, 1.0, "", va="top", ha="left", fontsize=7.2, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=0.9),
        )

        ax_btn_talbot = self.fig.add_subplot(gs_col3[2])
        self.btn_talbot = Button(ax_btn_talbot, "Show Talbot Carpet", color="#f1f5f9", hovercolor="#e2e8f0")
        self.btn_talbot.label.set_fontsize(8.5)
        self.btn_talbot.label.set_fontweight("bold")
        self.btn_talbot.label.set_color("#0f172a")
        self.btn_talbot.on_clicked(self.open_talbot_window)

        # 2x2 Subgridspec for Presets and Reset
        gs_btns = gs_col3[3].subgridspec(2, 2, hspace=0.25, wspace=0.15)

        self.btn_preset1 = Button(self.fig.add_subplot(gs_btns[0, 0]), "Single Slit", color="#f8fafc", hovercolor="#e2e8f0")
        self.btn_preset1.label.set_fontsize(7.5)
        self.btn_preset1.on_clicked(lambda _e: self.apply_preset("Single slit"))

        self.btn_preset2 = Button(self.fig.add_subplot(gs_btns[0, 1]), "Grating (CD)", color="#f8fafc", hovercolor="#e2e8f0")
        self.btn_preset2.label.set_fontsize(7.5)
        self.btn_preset2.on_clicked(lambda _e: self.apply_preset("Diffraction grating"))

        self.btn_preset3 = Button(self.fig.add_subplot(gs_btns[1, 0]), "Blazed Spec", color="#f8fafc", hovercolor="#e2e8f0")
        self.btn_preset3.label.set_fontsize(7.5)
        self.btn_preset3.on_clicked(lambda _e: self.apply_preset("Blazed spectrometer"))

        self.btn_reset = Button(self.fig.add_subplot(gs_btns[1, 1]), "Reset Defaults", color="#fee2e2", hovercolor="#fecaca")
        self.btn_reset.label.set_fontsize(7.5)
        self.btn_reset.label.set_fontweight("bold")
        self.btn_reset.label.set_color("#991b1b")
        self.btn_reset.on_clicked(self.reset)

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
            "Binary Amplitude": "Binary amplitude: chrome-on-glass slits",
            "Sinusoidal Phase": "Sinusoidal phase: continuous phase relief",
            "Blazed Phase": "Blazed phase: asymmetric sawtooth blazed grooves",
            "2D Orthogonal": "2D orthogonal: 2-axis square mesh",
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
            self.txt_wl.set_text("%0.0f" % wl)
        if theta is not None:
            self.s_theta.set_val(theta)
            self.txt_theta.set_text("%0.1f°" % theta)
        if waist is not None:
            self.s_waist.set_val(waist)
            self.txt_waist.set_text("%0.1f" % waist)
        if dist is not None:
            self.s_dist.set_val(dist)
            self.txt_dist.set_text("%0.2f m" % dist)
        if period is not None:
            self.s_period.set_val(period)
            self.txt_period.set_text("%0.2f" % period)
        if n_slits is not None:
            self.s_n.set_val(n_slits)
            self.txt_n.set_text("%d" % n_slits)
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
        """Perform exact wave-optics calculation, update plots, screen pattern, and info diagnostics."""
        self.wavelength = self.s_wl.val * 1e-9
        self.period = self.s_period.val * 1e-6
        self.duty = self.s_duty.val
        self.phase_depth = self.s_phase.val
        self.n_slits = int(self.s_n.val)
        self.theta_i = math.radians(self.s_theta.val)
        self.screen_dist = self.s_dist.val
        self.beam_waist = self.s_waist.val * 1e-6

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

        # 1. Aperture Plane Plot
        self.ax_profile.clear()
        x_um = grid.x * 1e6
        crop_mask = np.abs(grid.x) <= (4.0 * self.period)
        self.ax_profile.plot(x_um[crop_mask], np.abs(t_raw)[crop_mask], label="Magnitude |t|", color="#1f77b4", lw=1.5)
        self.ax_profile.plot(x_um[crop_mask], np.angle(t_raw)[crop_mask] / math.pi, label="Phase / π", color="#ff7f0e", ls="--", lw=1.2)
        self.ax_profile.set_title(f"Aperture Plane Transmittance: {self.grating_type}", fontsize=10, fontweight="bold", pad=16)
        self.ax_profile.set_xlabel("x [µm]", fontsize=8.5)
        self.ax_profile.set_ylabel("Amplitude / Phase", fontsize=8.5)
        self.ax_profile.grid(True, linestyle=":", alpha=0.5)
        self.ax_profile.legend(loc="upper right", fontsize=8, framealpha=0.8)

        # 2. Far-Field Fraunhofer Diffraction Plot
        sin_theta, intensity, theta_deg = propagation.far_field_fft(
            t_illuminated,
            dx=grid.dx,
            wavelength=self.wavelength,
            pad_factor=2,
        )

        self.ax_fft.clear()
        y_data = np.maximum(intensity, 1e-4) if self.log_scale else intensity
        self.ax_fft.plot(theta_deg, y_data, label="Numerical FFT", color="#2ca02c", lw=1.5)

        if self.compare_analytical and self.grating_type == "Binary Amplitude":
            theta_rad_dense = np.radians(theta_deg)
            i_analytical = analytical.n_slit_intensity(
                theta=theta_rad_dense,
                a=a,
                d=self.period,
                N=self.n_slits,
                wavelength=self.wavelength,
                theta_i=self.theta_i,
            )
            y_analytical = np.maximum(i_analytical, 1e-4) if self.log_scale else i_analytical
            self.ax_fft.plot(theta_deg, y_analytical, label="Analytical N-Slit", color="#d62728", ls=":", lw=1.3)

        orders, evanescent = analytical.order_angles(
            d=self.period,
            wavelength=self.wavelength,
            theta_i=self.theta_i,
            m_max=4,
        )
        for m, ang_rad in orders.items():
            ang_d = math.degrees(ang_rad)
            if -48.0 <= ang_d <= 48.0:
                self.ax_fft.axvline(ang_d, color="#718096", linestyle="--", alpha=0.5, lw=0.9)
                self.ax_fft.text(
                    ang_d,
                    1.01,
                    f"m={m:+d}" if m != 0 else "m=0",
                    transform=self.ax_fft.get_xaxis_transform(),
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    color="#334155",
                    fontweight="bold",
                )

        self.ax_fft.set_title("Far-Field Fraunhofer Diffraction Pattern", fontsize=10, fontweight="bold", pad=16)
        self.ax_fft.set_xlabel("Diffraction Angle θ [degrees]", fontsize=8.5)
        self.ax_fft.set_ylabel("Intensity (Log)" if self.log_scale else "Normalized Intensity", fontsize=8.5)
        self.ax_fft.set_xlim(-50.0, 50.0)
        if self.log_scale:
            self.ax_fft.set_yscale("log")
            self.ax_fft.set_ylim(1e-4, 1.2)
        else:
            self.ax_fft.set_ylim(-0.02, 1.08)
        self.ax_fft.grid(True, linestyle=":", alpha=0.5)
        self.ax_fft.legend(loc="upper right", fontsize=8, framealpha=0.8)

        # 3. Simulated Screen Pattern
        self.ax_screen.clear()
        screen_y_cm = propagation.sin_theta_to_screen_position(np.radians(theta_deg), self.screen_dist) * 100.0
        y_screen_linear = np.linspace(-30.0, 30.0, 600)
        i_screen = np.interp(y_screen_linear, screen_y_cm, intensity, left=0.0, right=0.0)
        strip_2d = np.tile(i_screen, (40, 1))

        laser_cmap = get_laser_colormap(self.s_wl.val)
        self.ax_screen.imshow(
            strip_2d,
            extent=[-30.0, 30.0, -1.5, 1.5],
            cmap=laser_cmap,
            aspect="auto",
            origin="lower",
        )
        self.ax_screen.set_title(
            f"Simulated Screen Pattern at L = {self.screen_dist:.2f} m (Laser: {int(self.s_wl.val)} nm)",
            fontsize=9,
            fontweight="bold",
            pad=4,
        )
        self.ax_screen.set_xlabel("Screen Position y [cm]", fontsize=8.5)
        self.ax_screen.set_yticks([])

        # 4. Monospace Info Diagnostics Panel
        missing = analytical.missing_orders(self.period, a) if self.grating_type == "Binary Amplitude" else []
        lines = [
            f"λ: {int(self.s_wl.val)} nm | d: {self.period*1e6:.2f} µm | θᵢ: {math.degrees(self.theta_i):.1f}°",
            f"N: {self.n_slits} slits | Screen L: {self.screen_dist:.2f} m",
            "─" * 30,
            "Propagating Orders:",
        ]
        for m in sorted(orders.keys()):
            lines.append(f" m = {m:+2d} : {math.degrees(orders[m]):+6.2f}°")

        if missing:
            lines.append("─" * 30)
            missing_str = ", ".join(f"±{m}" for m in missing[:4])
            lines.append(f"Missing: m = {missing_str}")
        if evanescent:
            lines.append(f"Evanescent: |m| ≥ {min(abs(e) for e in evanescent)}")

        self.info_text.set_text("\n".join(lines))
        self.fig.canvas.draw_idle()

    def open_talbot_window(self, _event=None) -> None:
        """Launch the near-field Talbot carpet simulation in a separate dedicated window."""
        fig_t, ax_t = plt.subplots(figsize=(9, 6))
        fig_t.canvas.manager.set_window_title("GratingLab: Near-Field Talbot Carpet")

        z_t = analytical.talbot_distance(self.period, self.wavelength)
        grid = gratings.Grid1D(width=20.0 * self.period, num_samples=1500)
        t_grating = gratings.binary_amplitude(grid.x, d=self.period, a=self.duty * self.period)

        x, z_arr, carpet = propagation.talbot_carpet(
            field=t_grating,
            dx=grid.dx,
            wavelength=self.wavelength,
            d=self.period,
            n_z_steps=200,
            z_max=2.0 * z_t,
        )

        crop = np.abs(x) <= (3.0 * self.period)
        im = ax_t.imshow(
            carpet[:, crop],
            extent=[x[crop][0] * 1e6, x[crop][-1] * 1e6, 0.0, z_arr[-1] * 1e6],
            aspect="auto",
            origin="lower",
            cmap="inferno",
        )
        ax_t.axhline(z_t * 1e6, color="cyan", ls="--", label=f"z_T = {z_t*1e6:.2f} µm")
        ax_t.set_title(f"Talbot Carpet ($d = {self.period*1e6:.2f}\\,\\mu\\mathrm{{m}}$, $\\lambda = {self.wavelength*1e9:.0f}\\,\\mathrm{{nm}}$)")
        ax_t.set_xlabel("Transverse x [µm]")
        ax_t.set_ylabel("Propagation Distance z [µm]")
        ax_t.legend(loc="upper right")
        fig_t.colorbar(im, ax=ax_t, label="Intensity")
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
