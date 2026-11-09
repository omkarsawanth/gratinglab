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
import time

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


def get_material_info(n: float) -> tuple[str, float]:
    """Return substrate material identification and normal-incidence Fresnel transmission."""
    if n <= 1.01:
        name = "Vacuum / Air"
    elif n < 1.36:
        name = "Water / Cryolite"
    elif n < 1.48:
        name = "Fused Silica"
    elif n < 1.54:
        name = "Crown Glass (BK7)"
    elif n < 1.62:
        name = "Polycarbonate"
    elif n < 1.72:
        name = "Dense Flint Glass"
    elif n < 1.85:
        name = "Sapphire (Al₂O₃)"
    else:
        name = "Diamond / High-Index"

    r_fresnel = ((n - 1.0) / (n + 1.0)) ** 2 if n >= 1.0 else 0.0
    t_fresnel = max(0.0, 1.0 - r_fresnel)
    return name, t_fresnel


class GratingLabApp:
    def __init__(
        self,
        grating_type: str = "Binary Amplitude",
        wavelength_nm: float = 650.0,
        period_um: float = 1.6,
        n_slits: int = 10,
        screen_dist_m: float = 0.5,
        n_index: float = 1.50,
    ) -> None:
        # Default snapshot for the Reset button
        self.default_params = {
            "grating_type": grating_type,
            "wavelength_nm": wavelength_nm,
            "period_um": period_um,
            "n_slits": n_slits,
            "screen_dist_m": screen_dist_m,
            "n_index": n_index,
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
        self.n_index = n_index
        self.duty = 0.5
        self.n_slits = n_slits
        self.theta_i = 0.0
        self.screen_dist = screen_dist_m
        self.beam_waist = 40.0e-6
        self.phase_depth = 2.0 * math.pi
        self.log_scale = False
        self.compare_analytical = True
        self.num_samples = 1024

        # Interactive state & performance guards
        self._bulk_updating = False
        self._last_sim_params = None
        self._last_log_scale = False
        self._active_preset_idx = None
        self._last_update_time = 0.0
        self._update_timer = None
        self._update_timer_running = False

        self.build_layout()
        self.build_controls()
        self.on_type(self.grating_type)

    def build_layout(self) -> None:
        """Create pure GridSpec dark scientific dashboard architecture without manual coordinates."""
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(16, 9.5), dpi=100, facecolor="#0b0f14")
        if hasattr(self.fig.canvas, "manager") and self.fig.canvas.manager is not None:
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

        # ─── Channel 1 Static Layout & Pre-instantiated Artists ───
        self.title_profile = self.ax_profile.set_title(
            f"CH1 // APERTURE TRANSMITTANCE [{self.grating_type.upper()}]",
            fontsize=9, fontweight="bold", color="#e6edf3", pad=10, loc="left"
        )
        self.ax_profile.set_xlabel("Transverse Coordinate x [µm]", fontsize=8, color="#8b949e", labelpad=2)
        self.ax_profile.set_ylabel("Field Amplitude / Phase (rad/π)", fontsize=8, color="#8b949e")
        self.ax_profile.set_ylim(-1.15, 1.28)
        self.ax_profile.grid(True, linestyle=":", color="#212936", alpha=0.8)

        self.poly_trans_fill = self.ax_profile.fill_between([0, 1], 0, [1, 1], color="#14b8a6", alpha=0.18, label="|t(x)| Magnitude")
        self.line_trans_mag, = self.ax_profile.plot([], [], color="#2dd4bf", lw=1.8)
        self.line_trans_phase, = self.ax_profile.plot([], [], color="#f97316", ls="--", lw=1.4, label="Phase φ(x) / π")
        self.line_trans_beam, = self.ax_profile.plot([], [], color="#94a3b8", ls=":", lw=1.2, alpha=0.75, label="Beam Envelope w₀")

        self.ann_period = self.ax_profile.annotate(
            "", xy=(0.0, 1.08), xytext=(1.6, 1.08),
            arrowprops=dict(arrowstyle="<->", color="#38bdf8", lw=1.0),
            visible=False,
        )
        self.ann_period_txt = self.ax_profile.text(
            0.8, 1.12, "",
            ha="center", va="bottom", fontsize=7.5, color="#38bdf8", family="monospace", fontweight="bold",
            visible=False,
        )

        leg1 = self.ax_profile.legend(loc="upper right", fontsize=7.2, facecolor="#161c24", edgecolor="#2c3543")
        for text in leg1.get_texts():
            text.set_color("#e6edf3")

        # ─── Channel 2 Static Layout & Pre-instantiated Artists ───
        self.title_fft = self.ax_fft.set_title(
            "CH2 // FAR-FIELD FRAUNHOFER DIFFRACTION SPECTRUM",
            fontsize=9, fontweight="bold", color="#e6edf3", pad=10, loc="left"
        )
        self.ax_fft.set_xlabel("Diffraction Angle θ [deg]", fontsize=8, color="#8b949e", labelpad=2)
        self.lbl_fft_ylabel = self.ax_fft.set_ylabel("Normalized Spectral Irradiance I(θ)", fontsize=8, color="#8b949e")
        self.ax_fft.set_xlim(-50.0, 50.0)
        self.ax_fft.set_ylim(-0.02, 1.08)
        self.ax_fft.grid(True, linestyle=":", color="#212936", alpha=0.8)

        self.poly_fft_fill = self.ax_fft.fill_between([0, 1], 0, [1, 1], color="#10b981", alpha=0.12)
        self.line_fft, = self.ax_fft.plot([], [], label="Numerical FFT", color="#10b981", lw=1.6)
        self.line_analytical, = self.ax_fft.plot([], [], label="Analytical N-Slit", color="#f59e0b", ls=":", lw=1.3, visible=False)
        self.line_envelope, = self.ax_fft.plot([], [], label="Slit Envelope sinc²(β)", color="#64748b", ls="-.", lw=1.1, alpha=0.8, visible=False)

        self.vline_orders_pool = [
            self.ax_fft.axvline(0, color="#334155", linestyle="--", alpha=0.6, lw=0.9, visible=False)
            for _ in range(30)
        ]
        self.txt_orders_pool = [
            self.ax_fft.text(
                0, 1.01, "",
                transform=self.ax_fft.get_xaxis_transform(),
                fontsize=8, ha="center", va="bottom",
                color="#38bdf8", fontweight="bold", family="monospace",
                visible=False,
            )
            for _ in range(30)
        ]

        self.scatter_peaks = self.ax_fft.scatter([], [], s=18, zorder=4, edgecolor="#0b0f14", linewidth=0.5)
        self.scatter_missing = self.ax_fft.scatter([], [], marker="x", color="#ef4444", s=22, zorder=5, lw=1.2)
        self.txt_missing_pool = [
            self.ax_fft.text(0, 0, "", ha="center", va="bottom", fontsize=6.8, color="#ef4444", family="monospace", visible=False)
            for _ in range(12)
        ]

        # ─── Detector Array Screen Strip ───
        self.ax_screen = self.fig.add_subplot(self.gs_main[2], facecolor="#000000")
        self.ax_screen.tick_params(colors="#8b949e", labelsize=8)
        for spine in self.ax_screen.spines.values():
            spine.set_color("#242c38")
            spine.set_linewidth(0.8)

        self.title_screen = self.ax_screen.set_title(
            "DETECTOR ARRAY // SIMULATED SCREEN PATTERN",
            fontsize=8.5, fontweight="bold", color="#e6edf3", pad=8, loc="left"
        )
        self.ax_screen.set_xlabel("Transverse Screen Displacement y [cm] (Metric Scale)", fontsize=8, color="#8b949e", labelpad=2)
        self.ax_screen.set_xlim(-30.0, 30.0)
        self.ax_screen.set_ylim(-1.5, 2.0)
        self.ax_screen.set_yticks([])

        self.im_screen = self.ax_screen.imshow(
            np.zeros((25, 400)),
            extent=[-30.0, 30.0, -1.5, 1.5],
            cmap="inferno",
            aspect="auto",
            origin="lower",
        )
        self.scatter_screen_pointers = self.ax_screen.scatter([], [], marker="v", color="#38bdf8", s=16, zorder=5)
        self.txt_screen_pool = [
            self.ax_screen.text(0, 1.48, "", ha="center", va="bottom", fontsize=6.8, color="#38bdf8", family="monospace", visible=False)
            for _ in range(30)
        ]

        # Deferred rate-limiting timer
        try:
            self._update_timer = self.fig.canvas.new_timer(interval=25)
            self._update_timer.single_shot = True
            self._update_timer.add_callback(self._on_deferred_update)
        except Exception:
            self._update_timer = None
        self._update_timer_running = False
        self._last_update_time = 0.0
        self._pending_update = False
        self._is_dragging = False
        self._bg_profile = None
        self._bg_fft = None
        self._bg_screen = None

        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

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
        """Construct a high-density, tactile laboratory slider with dedicated live readout."""
        gs_item = cell.subgridspec(2, 1, height_ratios=[0.42, 0.58], hspace=0.1)

        ax_lbl = self.fig.add_subplot(gs_item[0])
        ax_lbl.axis("off")
        ax_lbl.set_xticks([])
        ax_lbl.set_yticks([])
        ax_lbl.xaxis.set_visible(False)
        ax_lbl.yaxis.set_visible(False)
        ax_lbl.text(0.0, 0.1, title, fontsize=8.0, fontweight="bold", color="#8b949e", ha="left", va="bottom", family="monospace")
        val_txt = ax_lbl.text(1.0, 0.1, valfmt % valinit, fontsize=8.3, fontweight="bold", color=accent_color, ha="right", va="bottom", family="monospace")

        ax_track = self.fig.add_subplot(gs_item[1], facecolor="#161c24")
        ax_track.set_xticks([])
        ax_track.set_yticks([])
        ax_track.xaxis.set_visible(False)
        ax_track.yaxis.set_visible(False)
        for spine in ax_track.spines.values():
            spine.set_color("#242c38")
            spine.set_linewidth(0.8)

        slider_kwargs = {
            "valinit": valinit,
            "valstep": valstep,
            "color": accent_color,
            "track_color": "#161c24",
            "initcolor": "none",
        }
        try:
            slider = Slider(
                ax_track,
                "",
                valmin,
                valmax,
                handle_style={"facecolor": "#ffffff", "edgecolor": accent_color, "size": 9},
                **slider_kwargs,
            )
        except TypeError:
            slider = Slider(ax_track, "", valmin, valmax, **slider_kwargs)

        slider.valtext.set_visible(False)
        slider.drawon = False

        def on_change(val):
            self._is_dragging = True
            val_txt.set_text(valfmt % val)
            if hasattr(self.fig.canvas, "copy_from_bbox"):
                try:
                    ax_track.draw_artist(slider.poly)
                    if hasattr(slider, "_handle"):
                        ax_track.draw_artist(slider._handle)
                    self.fig.canvas.blit(ax_track.bbox)
                    ax_lbl.draw_artist(val_txt)
                    self.fig.canvas.blit(ax_lbl.bbox)
                except Exception:
                    pass
            if not self._bulk_updating:
                if self._active_preset_idx is not None:
                    self._set_active_preset(None)
                self._request_update()

        slider.on_changed(on_change)
        return slider, ax_lbl, ax_track, val_txt

    def _on_draw(self, event=None) -> None:
        """Cache clean static backgrounds for fast blitting."""
        if not hasattr(self.fig.canvas, "copy_from_bbox"):
            return
        try:
            if not self._is_dragging:
                self._bg_profile = self.fig.canvas.copy_from_bbox(self.ax_profile.bbox)
                self._bg_fft = self.fig.canvas.copy_from_bbox(self.ax_fft.bbox)
                self._bg_screen = self.fig.canvas.copy_from_bbox(self.ax_screen.bbox)
        except Exception:
            self._bg_profile = None
            self._bg_fft = None
            self._bg_screen = None

    def _on_resize(self, event=None) -> None:
        """Invalidate blit background cache on window resize."""
        self._bg_profile = None
        self._bg_fft = None
        self._bg_screen = None

    def _on_release(self, event=None) -> None:
        """Ensure full figure redraw when mouse interaction concludes."""
        if self._is_dragging:
            self._is_dragging = False
            self.update(None, full_draw=True)

    def _fast_draw(self) -> bool:
        """Fast blit update of simulation oscilloscopes and detector screen. Returns True if blit succeeded."""
        if self._bg_profile is None or self._bg_fft is None or self._bg_screen is None:
            return False

        canvas = self.fig.canvas
        try:
            canvas.restore_region(self._bg_profile)
            canvas.restore_region(self._bg_fft)
            canvas.restore_region(self._bg_screen)

            # Channel 1: Aperture transmittance
            self.ax_profile.draw_artist(self.poly_trans_fill)
            self.ax_profile.draw_artist(self.line_trans_mag)
            self.ax_profile.draw_artist(self.line_trans_phase)
            self.ax_profile.draw_artist(self.line_trans_beam)
            if self.ann_period.get_visible():
                self.ax_profile.draw_artist(self.ann_period)
                self.ax_profile.draw_artist(self.ann_period_txt)

            # Channel 2: Fraunhofer FFT
            self.ax_fft.draw_artist(self.poly_fft_fill)
            self.ax_fft.draw_artist(self.line_fft)
            if self.line_analytical.get_visible():
                self.ax_fft.draw_artist(self.line_analytical)
                self.ax_fft.draw_artist(self.line_envelope)

            for vl in self.vline_orders_pool:
                if vl.get_visible():
                    self.ax_fft.draw_artist(vl)

            for txt in self.txt_orders_pool:
                if txt.get_visible():
                    self.ax_fft.draw_artist(txt)

            if self.scatter_peaks.get_visible():
                self.ax_fft.draw_artist(self.scatter_peaks)

            if self.scatter_missing.get_visible():
                self.ax_fft.draw_artist(self.scatter_missing)

            for txt in self.txt_missing_pool:
                if txt.get_visible():
                    self.ax_fft.draw_artist(txt)

            # Channel 3: Detector Screen
            self.ax_screen.draw_artist(self.im_screen)
            if self.scatter_screen_pointers.get_visible():
                self.ax_screen.draw_artist(self.scatter_screen_pointers)

            for txt in self.txt_screen_pool:
                if txt.get_visible():
                    self.ax_screen.draw_artist(txt)

            canvas.blit(self.ax_profile.bbox)
            canvas.blit(self.ax_fft.bbox)
            canvas.blit(self.ax_screen.bbox)
            return True
        except Exception:
            return False

    def _on_deferred_update(self) -> None:
        """Callback fired by single-shot timer to ensure final position updates."""
        self._do_throttled_update()

    def _do_throttled_update(self) -> None:
        if not self._pending_update:
            return
        self._pending_update = False
        self._update_timer_running = False
        try:
            self.update(None, fast_mode=self._is_dragging)
        finally:
            self._last_update_time = time.perf_counter()

    def _request_update(self) -> None:
        """Rate-limit simulation updates to ~30 FPS to prevent GUI event-loop choking."""
        self._pending_update = True
        # In non-interactive/headless environments (e.g. testing with Agg), update immediately
        if self._update_timer is None or matplotlib.get_backend().lower() in ("agg", "template", "pdf", "ps", "svg"):
            self._pending_update = False
            self.update(None)
            return

        if self._update_timer_running:
            return

        now = time.perf_counter()
        elapsed = now - self._last_update_time
        if elapsed >= 0.035:
            self._do_throttled_update()
        else:
            self._update_timer_running = True
            try:
                self._update_timer.interval = max(10, int((0.035 - elapsed) * 1000))
                self._update_timer.start()
            except Exception:
                self._update_timer_running = False
                self._do_throttled_update()

    def _update_fft_legend(self) -> None:
        """Update Channel 2 legend without destroying the Axes or incurring layout rebuilds."""
        if self.compare_analytical and self.grating_type == "Binary Amplitude":
            handles = [self.line_fft, self.line_analytical, self.line_envelope]
        else:
            handles = [self.line_fft]
        self.leg_fft = self.ax_fft.legend(
            handles=handles,
            loc="upper right",
            fontsize=7.2,
            facecolor="#161c24",
            edgecolor="#2c3543",
        )
        for text in self.leg_fft.get_texts():
            text.set_color("#e6edf3")

    def _set_active_preset(self, active_idx: int | None) -> None:
        """Update top preset buttons to indicate selected active preset state."""
        if self._active_preset_idx == active_idx:
            return
        self._active_preset_idx = active_idx
        preset_btns = [self.btn_preset1, self.btn_preset2, self.btn_preset3]
        for i, btn in enumerate(preset_btns):
            if active_idx is not None and i == active_idx:
                btn.color = "#0f4a47"
                btn.hovercolor = "#115e59"
                btn.ax.set_facecolor("#0f4a47")
                btn.label.set_color("#2dd4bf")
                btn.label.set_fontweight("bold")
                for spine in btn.ax.spines.values():
                    spine.set_color("#2dd4bf")
                    spine.set_linewidth(1.2)
            else:
                btn.color = "#161c24"
                btn.hovercolor = "#242c38"
                btn.ax.set_facecolor("#161c24")
                btn.label.set_color("#8b949e")
                btn.label.set_fontweight("normal")
                for spine in btn.ax.spines.values():
                    spine.set_color("#242c38")
                    spine.set_linewidth(0.8)
        self.fig.canvas.draw_idle()

    def _update_radio_styling(self) -> None:
        """Update radio buttons to highlight the active grating profile."""
        types = ["Binary Amplitude", "Sinusoidal Phase", "Blazed Phase", "2D Orthogonal"]
        active_idx = types.index(self.grating_type) if self.grating_type in types else 0
        for i, lbl in enumerate(self.radio_type.labels):
            if i == active_idx:
                lbl.set_color("#2dd4bf")
                lbl.set_fontweight("bold")
                lbl.set_fontsize(8.2)
            else:
                lbl.set_color("#8b949e")
                lbl.set_fontweight("normal")
                lbl.set_fontsize(8.0)
        try:
            edge_colors = ["#14b8a6" if i == active_idx else "#242c38" for i in range(len(types))]
            face_colors = ["#042f2e" if i == active_idx else "#161c24" for i in range(len(types))]
            self.radio_type.set_radio_props({"edgecolor": edge_colors, "facecolor": face_colors})
        except Exception:
            pass

    def _update_check_styling(self) -> None:
        """Update check buttons to illuminate active visualization toggles."""
        statuses = [self.log_scale, self.compare_analytical]
        for i, lbl in enumerate(self.check_toggles.labels):
            if statuses[i]:
                lbl.set_color("#38bdf8")
                lbl.set_fontweight("bold")
            else:
                lbl.set_color("#64748b")
                lbl.set_fontweight("normal")
        try:
            edge_colors = ["#38bdf8" if s else "#242c38" for s in statuses]
            face_colors = ["#082f49" if s else "#161c24" for s in statuses]
            check_colors = ["#38bdf8" if s else "#242c38" for s in statuses]
            self.check_toggles.set_frame_props({"edgecolor": edge_colors, "facecolor": face_colors})
            self.check_toggles.set_check_props({"color": check_colors})
        except Exception:
            pass

    def build_controls(self) -> None:
        """Construct the top preset bar, 3 instrument control decks, and real-time telemetry terminal."""
        def _clean_control_ax(ax):
            ax.set_xticks([])
            ax.set_yticks([])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            return ax

        # ─────────────────────────────────────────────────────────────
        # Top Header Bar Content
        # ─────────────────────────────────────────────────────────────
        _clean_control_ax(self.ax_head_brand)
        self.ax_head_brand.text(0.0, 0.85, "GRATINGLAB // OPTICAL SPECTROMETER", fontsize=10.5, fontweight="bold", color="#f0f6fc", family="sans-serif")
        self.ax_head_brand.text(0.0, 0.15, "SYS-ID: GL-650X-PRO  |  PRECISION OPTICAL METROLOGY", fontsize=7.6, color="#8b949e", family="monospace")

        _clean_control_ax(self.ax_head_status)
        self.lbl_head_status = self.ax_head_status.text(0.5, 0.85, "● SYSTEM ONLINE", fontsize=8.8, fontweight="bold", color="#10b981", family="monospace", ha="center")
        self.lbl_head_telemetry = self.ax_head_status.text(0.5, 0.15, "LIVE // λ = 650 nm | d = 1.60 µm | n = 1.50", fontsize=7.6, color="#64748b", family="monospace", ha="center")

        # Top Instrument Action Buttons
        ax_btn1 = _clean_control_ax(self.fig.add_subplot(self.gs_hbtns[0]))
        self.btn_preset1 = Button(ax_btn1, "Single Slit", color="#161c24", hovercolor="#242c38")
        self.btn_preset1.label.set_fontsize(7.5)
        self.btn_preset1.label.set_color("#e6edf3")
        self.btn_preset1.on_clicked(lambda _e: self.apply_preset("Single slit"))

        ax_btn2 = _clean_control_ax(self.fig.add_subplot(self.gs_hbtns[1]))
        self.btn_preset2 = Button(ax_btn2, "Grating (CD)", color="#161c24", hovercolor="#242c38")
        self.btn_preset2.label.set_fontsize(7.5)
        self.btn_preset2.label.set_color("#e6edf3")
        self.btn_preset2.on_clicked(lambda _e: self.apply_preset("Diffraction grating"))

        ax_btn3 = _clean_control_ax(self.fig.add_subplot(self.gs_hbtns[2]))
        self.btn_preset3 = Button(ax_btn3, "Blazed Spec", color="#161c24", hovercolor="#242c38")
        self.btn_preset3.label.set_fontsize(7.5)
        self.btn_preset3.label.set_color("#e6edf3")
        self.btn_preset3.on_clicked(lambda _e: self.apply_preset("Blazed spectrometer"))

        ax_btn4 = _clean_control_ax(self.fig.add_subplot(self.gs_hbtns[3]))
        self.btn_talbot = Button(ax_btn4, "Talbot 2D", color="#064e3b", hovercolor="#047857")
        self.btn_talbot.label.set_fontsize(7.5)
        self.btn_talbot.label.set_fontweight("bold")
        self.btn_talbot.label.set_color("#34d399")
        self.btn_talbot.on_clicked(self.open_talbot_window)

        ax_btn5 = _clean_control_ax(self.fig.add_subplot(self.gs_hbtns[4]))
        self.btn_reset = Button(ax_btn5, "Reset", color="#2c1517", hovercolor="#4c1d24")
        self.btn_reset.label.set_fontsize(7.5)
        self.btn_reset.label.set_fontweight("bold")
        self.btn_reset.label.set_color("#fca5a5")
        self.btn_reset.on_clicked(self.reset)

        # ─────────────────────────────────────────────────────────────
        # Section 01: Grating Profile & View Modes (Theme: Sky / Teal)
        # ─────────────────────────────────────────────────────────────
        gs_c0 = self.gs_controls[0].subgridspec(5, 1, height_ratios=[0.38, 1.7, 0.35, 0.95, 0.58], hspace=0.22)

        ax_h0 = _clean_control_ax(self.fig.add_subplot(gs_c0[0]))
        ax_h0.axis("off")
        ax_h0.text(0.0, 0.85, "SECTION 01 // GRATING PROFILE", fontsize=8.5, fontweight="bold", color="#38bdf8", family="monospace")
        ax_h0.text(0.0, 0.1, "Aperture modulation function", fontsize=7.5, color="#64748b")

        ax_radio = _clean_control_ax(self.fig.add_subplot(gs_c0[1], facecolor="#11161d"))
        for spine in ax_radio.spines.values():
            spine.set_color("#242c38")
            spine.set_linewidth(0.8)

        self.radio_type = RadioButtons(
            ax_radio,
            ("Binary Amplitude", "Sinusoidal Phase", "Blazed Phase", "2D Orthogonal"),
            active=["Binary Amplitude", "Sinusoidal Phase", "Blazed Phase", "2D Orthogonal"].index(self.grating_type),
            activecolor="#14b8a6",
        )
        self.radio_type.on_clicked(self.on_type)
        self._update_radio_styling()

        ax_h0_sub = _clean_control_ax(self.fig.add_subplot(gs_c0[2]))
        ax_h0_sub.axis("off")
        ax_h0_sub.text(0.0, 0.1, "VIEW MODES:", fontsize=8, fontweight="bold", color="#8b949e", family="monospace")

        ax_checks = _clean_control_ax(self.fig.add_subplot(gs_c0[3], facecolor="#11161d"))
        for spine in ax_checks.spines.values():
            spine.set_color("#242c38")
            spine.set_linewidth(0.8)

        self.check_toggles = CheckButtons(
            ax_checks,
            ("Log Scale", "Compare Analytical"),
            (self.log_scale, self.compare_analytical),
        )
        self.check_toggles.on_clicked(self.on_toggle)
        self._update_check_styling()

        ax_p_hint = _clean_control_ax(self.fig.add_subplot(gs_c0[4]))
        ax_p_hint.axis("off")
        self.lbl_profile_telemetry = ax_p_hint.text(
            0.0, 0.9, "",
            fontsize=6.8, color="#64748b", family="monospace", va="top", linespacing=1.22
        )

        # ─────────────────────────────────────────────────────────────
        # Section 02: Laser Emitter (Theme: Laboratory Orange #f97316)
        # ─────────────────────────────────────────────────────────────
        gs_c1 = self.gs_controls[1].subgridspec(6, 1, height_ratios=[0.38, 0.82, 0.82, 0.82, 0.82, 0.58], hspace=0.25)

        ax_h1 = _clean_control_ax(self.fig.add_subplot(gs_c1[0]))
        ax_h1.axis("off")
        ax_h1.text(0.0, 0.85, "SECTION 02 // LASER EMITTER", fontsize=8.5, fontweight="bold", color="#f97316", family="monospace")
        ax_h1.text(0.0, 0.1, "Coherent laser source excitation", fontsize=7.5, color="#64748b")

        self.s_wl, _, _, self.txt_wl = self._create_dark_slider(
            gs_c1[1], "WAVELENGTH λ", 400.0, 700.0, self.wavelength * 1e9, 1.0, "%0.0f nm", "#f97316"
        )
        self.s_theta, _, _, self.txt_theta = self._create_dark_slider(
            gs_c1[2], "INCIDENCE ANGLE θᵢ", -30.0, 30.0, math.degrees(self.theta_i), 0.5, "%+0.1f°", "#f97316"
        )
        self.s_waist, _, _, self.txt_waist = self._create_dark_slider(
            gs_c1[3], "BEAM WAIST w₀", 5.0, 100.0, self.beam_waist * 1e6, 1.0, "%0.1f µm", "#f97316"
        )
        self.s_dist, _, _, self.txt_dist = self._create_dark_slider(
            gs_c1[4], "SCREEN DISTANCE L", 0.10, 2.00, self.screen_dist, 0.02, "%0.2f m", "#f97316"
        )

        ax_l_telemetry = _clean_control_ax(self.fig.add_subplot(gs_c1[5]))
        ax_l_telemetry.axis("off")
        self.lbl_laser_telemetry = ax_l_telemetry.text(
            0.0, 0.9, "",
            fontsize=6.8, color="#fdba74", family="monospace", va="top", linespacing=1.22
        )

        # ─────────────────────────────────────────────────────────────
        # Section 03: Grating Microstructure & Material (Theme: Teal #14b8a6)
        # ─────────────────────────────────────────────────────────────
        gs_c2 = self.gs_controls[2].subgridspec(6, 1, height_ratios=[0.38, 0.82, 0.82, 0.82, 0.82, 0.58], hspace=0.25)

        ax_h2 = _clean_control_ax(self.fig.add_subplot(gs_c2[0]))
        ax_h2.axis("off")
        ax_h2.text(0.0, 0.85, "SECTION 03 // GRATING STRUCTURE", fontsize=8.5, fontweight="bold", color="#14b8a6", family="monospace")
        ax_h2.text(0.0, 0.1, "Microstructure pitch & substrate material", fontsize=7.5, color="#64748b")

        self.s_period, _, _, self.txt_period = self._create_dark_slider(
            gs_c2[1], "GRATING PITCH d", 0.50, 20.00, self.period * 1e6, 0.05, "%0.2f µm", "#14b8a6"
        )
        self.s_nindex, _, _, self.txt_nindex = self._create_dark_slider(
            gs_c2[2], "REFRACTIVE INDEX n", 1.00, 2.50, self.n_index, 0.01, "%0.2f", "#14b8a6"
        )
        self.s_n, _, _, self.txt_n = self._create_dark_slider(
            gs_c2[3], "RULINGS COUNT N", 2, 50, self.n_slits, 1, "%d lines", "#14b8a6"
        )

        # Progressive disclosure slot (row 4): Duty vs Phase Depth
        self.s_duty, self.ax_duty_lbl, self.ax_duty_track, self.txt_duty = self._create_dark_slider(
            gs_c2[4], "DUTY CYCLE / SLIT γ", 0.05, 0.95, self.duty, 0.01, "%0.2f", "#14b8a6"
        )
        self.s_phase, self.ax_phase_lbl, self.ax_phase_track, self.txt_phase = self._create_dark_slider(
            gs_c2[4], "PHASE DEPTH φ₀", 0.10, 4.0 * math.pi, self.phase_depth, 0.05, "%0.2f rad", "#14b8a6"
        )

        ax_g_telemetry = _clean_control_ax(self.fig.add_subplot(gs_c2[5]))
        ax_g_telemetry.axis("off")
        self.lbl_grating_telemetry = ax_g_telemetry.text(
            0.0, 0.9, "",
            fontsize=6.8, color="#5eead4", family="monospace", va="top", linespacing=1.22
        )

        # ─────────────────────────────────────────────────────────────
        # Section 04: Bloomberg-Grade Telemetry Terminal (Theme: Cyan #38bdf8)
        # ─────────────────────────────────────────────────────────────
        gs_c3 = self.gs_controls[3].subgridspec(2, 1, height_ratios=[0.38, 3.86], hspace=0.08)

        ax_h3 = _clean_control_ax(self.fig.add_subplot(gs_c3[0]))
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
            0.04, 0.95, "",
            va="top", ha="left", fontsize=7.1, family="monospace", color="#7dd3fc",
            linespacing=1.25
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

        self._update_radio_styling()
        self._update_fft_legend()

        profile_hints = {
            "Binary Amplitude": "Mechanism: Spatial amplitude mask\nSubstrate: Transmissive optical window",
            "Sinusoidal Phase": "Mechanism: Harmonic OPD phase delay\nDiffraction: Bessel J_m distribution",
            "Blazed Phase": "Mechanism: Asymmetric blazed groove\nBlaze Peak: Directed order focus",
            "2D Orthogonal": "Mechanism: 2-axis orthogonal grid\nDiffraction: 2D spatial lattice",
        }
        self.lbl_profile_telemetry.set_text(profile_hints.get(label, ""))
        if not self._bulk_updating:
            self._set_active_preset(None)
            self.update(None)

    def on_toggle(self, _label: str) -> None:
        """Handle view check box toggles."""
        status = self.check_toggles.get_status()
        self.log_scale = status[0]
        self.compare_analytical = status[1]
        self._update_check_styling()
        self._update_fft_legend()
        if not self._bulk_updating:
            self._set_active_preset(None)
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
            n_index=d.get("n_index", 1.50),
            log=d["log_scale"],
            comp=d["compare_analytical"],
        )
        self._set_active_preset(None)
        self.lbl_head_status.set_text("● RESET RESTORED")
        self.lbl_head_status.set_color("#fca5a5")

    def apply_preset(self, name: str) -> None:
        """Apply pre-configured optical presets in a single atomic update."""
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
                n_index=1.50,
            )
            self._set_active_preset(0)
            self.lbl_head_status.set_text("● PRESET: SINGLE SLIT")
            self.lbl_head_status.set_color("#2dd4bf")
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
                n_index=1.50,
            )
            self._set_active_preset(1)
            self.lbl_head_status.set_text("● PRESET: GRATING (CD)")
            self.lbl_head_status.set_color("#2dd4bf")
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
                n_index=1.50,
            )
            self._set_active_preset(2)
            self.lbl_head_status.set_text("● PRESET: BLAZED SPEC")
            self.lbl_head_status.set_color("#2dd4bf")

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
        n_index: float | None = None,
        log: bool | None = None,
        comp: bool | None = None,
    ) -> None:
        """Programmatically update widgets and state in a single bulk operation."""
        self._bulk_updating = True
        try:
            sliders = [
                self.s_wl, self.s_theta, self.s_waist, self.s_dist,
                self.s_period, self.s_nindex, self.s_n, self.s_duty, self.s_phase
            ]
            for s in sliders:
                s.eventson = False

            if wl is not None:
                self.s_wl.set_val(wl)
                color_name = "Red" if wl >= 620 else ("Green" if wl >= 495 else "Blue/Violet")
                self.txt_wl.set_text(f"{wl:.0f} nm [{color_name}]")
            if theta is not None:
                self.s_theta.set_val(theta)
                self.txt_theta.set_text(f"{theta:+.1f}°")
            if waist is not None:
                self.s_waist.set_val(waist)
                self.txt_waist.set_text(f"{waist:.1f} µm")
            if dist is not None:
                self.s_dist.set_val(dist)
                self.txt_dist.set_text(f"{dist:.2f} m")
            if period is not None:
                self.s_period.set_val(period)
                lines_per_mm = 1e3 / period
                self.txt_period.set_text(f"{period:.2f} µm ({lines_per_mm:.0f} l/mm)")
            if n_index is not None:
                self.s_nindex.set_val(n_index)
                mat_name, _ = get_material_info(n_index)
                self.txt_nindex.set_text(f"{n_index:.2f} ({mat_name.split()[0]})")
            if n_slits is not None:
                self.s_n.set_val(n_slits)
                self.txt_n.set_text(f"{n_slits:d} lines")
            if duty is not None:
                self.s_duty.set_val(duty)
                p_val = self.s_period.val
                self.txt_duty.set_text(f"{duty:.2f} (a={duty*p_val:.2f}µm)")
            if phase is not None:
                self.s_phase.set_val(phase)
                n_val = self.s_nindex.val
                wl_val = self.s_wl.val
                h_nm = (phase * wl_val) / (math.pi * (n_val - 1.0)) if n_val > 1.0 else 0.0
                self.txt_phase.set_text(f"{phase/math.pi:.2f}π rad (h={h_nm:.0f}nm)")

            for s in sliders:
                s.eventson = True

            if grating_type is not None:
                self.grating_type = grating_type
                types = ["Binary Amplitude", "Sinusoidal Phase", "Blazed Phase", "2D Orthogonal"]
                if grating_type in types:
                    self.radio_type.eventson = False
                    self.radio_type.set_active(types.index(grating_type))
                    self.radio_type.eventson = True
                    self._update_radio_styling()

            if log is not None or comp is not None:
                self.check_toggles.eventson = False
                if log is not None and log != self.log_scale:
                    self.check_toggles.set_active(0)
                if comp is not None and comp != self.compare_analytical:
                    self.check_toggles.set_active(1)
                self.check_toggles.eventson = True
                self._update_check_styling()

        finally:
            self._bulk_updating = False

        self.on_type(self.grating_type)

    def update(self, _val=None, full_draw: bool = False, fast_mode: bool = False) -> None:
        """Perform exact wave-optics calculation, update research-grade plots, and telemetry terminal."""
        if self._bulk_updating:
            return

        wl_nm = self.s_wl.val
        theta_deg = self.s_theta.val
        waist_um = self.s_waist.val
        dist_m = self.s_dist.val
        period_um = self.s_period.val
        n_idx = self.s_nindex.val
        n_slits_val = int(round(self.s_n.val))
        duty_val = self.s_duty.val
        phase_val = self.s_phase.val

        current_sim_params = (
            self.grating_type,
            round(wl_nm, 1),
            round(theta_deg, 2),
            round(waist_um, 1),
            round(dist_m, 2),
            round(period_um, 3),
            round(n_idx, 2),
            n_slits_val,
            round(duty_val, 2),
            round(phase_val, 2),
            self.compare_analytical,
        )

        # Update dynamic live readout strings with physical feedback
        color_name = "Red" if wl_nm >= 620 else ("Green" if wl_nm >= 495 else "Blue/Violet")
        self.txt_wl.set_text(f"{wl_nm:.0f} nm [{color_name}]")
        self.txt_theta.set_text(f"{theta_deg:+.1f}°")
        self.txt_waist.set_text(f"{waist_um:.1f} µm")
        self.txt_dist.set_text(f"{dist_m:.2f} m")

        lines_per_mm = 1e3 / period_um if period_um > 0 else 0.0
        self.txt_period.set_text(f"{period_um:.2f} µm ({lines_per_mm:.0f} l/mm)")

        mat_name, t_fresnel = get_material_info(n_idx)
        self.txt_nindex.set_text(f"{n_idx:.2f} ({mat_name.split()[0]})")
        self.txt_n.set_text(f"{n_slits_val:d} lines")

        slit_width_um = duty_val * period_um
        self.txt_duty.set_text(f"{duty_val:.2f} (a={slit_width_um:.2f}µm)")

        h_relief_nm = (phase_val * wl_nm) / (math.pi * (n_idx - 1.0)) if n_idx > 1.0 else 0.0
        self.txt_phase.set_text(f"{phase_val/math.pi:.2f}π rad (h={h_relief_nm:.0f}nm)")

        # Update bottom telemetry hints
        freq_thz = (3e8 / (wl_nm * 1e-9)) * 1e-12
        energy_ev = (6.626e-34 * 3e8 / (wl_nm * 1e-9)) / 1.602e-19
        wavenumber_um = 1.0 / (wl_nm * 1e-3)
        self.lbl_laser_telemetry.set_text(
            f"OPTICAL BAND : {wl_nm:.0f} nm [{color_name}] | ν = {freq_thz:.1f} THz\n"
            f"QUANTUM BEAM : E = {energy_ev:.2f} eV | k = {wavenumber_um:.2f} µm⁻¹"
        )

        is_phase = self.grating_type in ("Sinusoidal Phase", "Blazed Phase")
        if is_phase:
            self.lbl_grating_telemetry.set_text(
                f"MATERIAL: {mat_name} (n = {n_idx:.2f})\n"
                f"RELIEF  : {lines_per_mm:.0f} l/mm  |  Groove h = {h_relief_nm:.0f} nm"
            )
        else:
            self.lbl_grating_telemetry.set_text(
                f"SUBSTRATE: {mat_name} (n = {n_idx:.2f})\n"
                f"GEOMETRY : {lines_per_mm:.0f} l/mm  |  Slit a = {slit_width_um:.2f} µm"
            )

        # Update top status telemetry
        self.lbl_head_telemetry.set_text(f"LIVE // λ = {wl_nm:.0f} nm | d = {period_um:.2f} µm | n = {n_idx:.2f}")

        # Optimization: Check if simulation recalculation is actually required
        if current_sim_params == self._last_sim_params:
            if self.log_scale != self._last_log_scale:
                self._last_log_scale = self.log_scale
                self.ax_fft.set_yscale("log" if self.log_scale else "linear")
                if self.log_scale:
                    self.ax_fft.set_ylim(1e-4, 1.25)
                else:
                    self.ax_fft.set_ylim(-0.02, 1.08)
                self.fig.canvas.draw_idle()
            return

        self._last_sim_params = current_sim_params
        self._last_log_scale = self.log_scale

        self.wavelength = wl_nm * 1e-9
        self.period = period_um * 1e-6
        self.n_index = n_idx
        self.duty = duty_val
        self.phase_depth = phase_val
        self.n_slits = n_slits_val
        self.theta_i = math.radians(theta_deg)
        self.screen_dist = dist_m
        self.beam_waist = waist_um * 1e-6

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
        x_um = grid.x * 1e6
        crop_mask = np.abs(grid.x) <= (4.0 * self.period)
        xc = x_um[crop_mask]
        t_mag = np.abs(t_raw)[crop_mask]
        t_phase = np.angle(t_raw)[crop_mask] / math.pi
        beam_mag = np.abs(incident_beam)[crop_mask]
        d_um = self.period * 1e6

        verts_prof = np.column_stack([np.concatenate([xc, xc[::-1]]), np.concatenate([np.zeros_like(t_mag), t_mag[::-1]])])
        self.poly_trans_fill.set_verts([verts_prof])

        self.line_trans_mag.set_data(xc, t_mag)
        self.line_trans_phase.set_data(xc, t_phase)
        self.line_trans_beam.set_data(xc, beam_mag)
        self.ax_profile.set_xlim(xc[0], xc[-1])

        # Period dimension annotation bracket
        if 0.5 <= d_um <= (np.sum(crop_mask) * (grid.dx * 1e6)):
            self.ann_period.xy = (0.0, 1.08)
            self.ann_period.set_position((d_um, 1.08))
            self.ann_period.set_visible(True)
            self.ann_period_txt.set_position((d_um / 2.0, 1.12))
            self.ann_period_txt.set_text(f"d = {d_um:.2f} µm")
            self.ann_period_txt.set_visible(True)
        else:
            self.ann_period.set_visible(False)
            self.ann_period_txt.set_visible(False)

        self.title_profile.set_text(f"CH1 // APERTURE TRANSMITTANCE [{self.grating_type.upper()}]")

        # ─── 2. Channel 2: Far-Field Fraunhofer Diffraction Spectrum ───
        sin_theta, intensity, theta_deg = propagation.far_field_fft(
            t_illuminated,
            dx=grid.dx,
            wavelength=self.wavelength,
            pad_factor=2,
        )
        theta_rad_dense = np.radians(theta_deg)
        y_data = np.maximum(intensity, 1e-4) if self.log_scale else intensity

        base_y = 1e-4 if self.log_scale else 0.0
        verts_fft = np.column_stack([np.concatenate([theta_deg, theta_deg[::-1]]), np.concatenate([np.full_like(y_data, base_y), y_data[::-1]])])
        self.poly_fft_fill.set_verts([verts_fft])
        self.line_fft.set_data(theta_deg, y_data)

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
            self.line_analytical.set_data(theta_deg, y_analytical)
            self.line_analytical.set_visible(True)

            # Single-slit diffraction envelope sinc²(β)
            beta_norm = (a / self.wavelength) * np.sin(theta_rad_dense)
            sinc_sq = np.sinc(beta_norm)**2
            y_env = np.maximum(sinc_sq, 1e-4) if self.log_scale else sinc_sq
            self.line_envelope.set_data(theta_deg, y_env)
            self.line_envelope.set_visible(True)
        else:
            self.line_analytical.set_visible(False)
            self.line_envelope.set_visible(False)

        # Propagating diffraction orders
        orders, evanescent = analytical.order_angles(
            d=self.period,
            wavelength=self.wavelength,
            theta_i=self.theta_i,
            m_max=4,
        )
        order_items = sorted(orders.items(), key=lambda it: it[0])
        idx_o = 0
        peak_pts = []
        marker_colors = []

        for m, ang_rad in order_items:
            ang_d = math.degrees(ang_rad)
            if -48.0 <= ang_d <= 48.0:
                if idx_o < len(self.vline_orders_pool):
                    vl = self.vline_orders_pool[idx_o]
                    vl.set_xdata([ang_d, ang_d])
                    vl.set_visible(True)

                    txt = self.txt_orders_pool[idx_o]
                    txt.set_position((ang_d, 1.01))
                    txt.set_text(f"m={m:+d}" if m != 0 else "m=0")
                    txt.set_visible(True)
                    idx_o += 1

                # Diffraction peak tip marker
                peak_idx = np.argmin(np.abs(theta_deg - ang_d))
                peak_val = y_data[peak_idx]
                if peak_val > (1e-3 if self.log_scale else 0.05):
                    col = "#f97316" if (self.grating_type == "Blazed Phase" and m == 1) else "#38bdf8"
                    peak_pts.append([ang_d, peak_val])
                    marker_colors.append(col)

        for j in range(idx_o, len(self.vline_orders_pool)):
            self.vline_orders_pool[j].set_visible(False)
            self.txt_orders_pool[j].set_visible(False)

        if peak_pts:
            self.scatter_peaks.set_offsets(np.array(peak_pts))
            self.scatter_peaks.set_color(marker_colors)
            self.scatter_peaks.set_visible(True)
        else:
            self.scatter_peaks.set_offsets(np.empty((0, 2)))
            self.scatter_peaks.set_visible(False)

        # Suppressed missing orders markers (e.g. m = ±2)
        missing_pts = []
        missing_labels = []
        if self.grating_type == "Binary Amplitude":
            missing_list = analytical.missing_orders(self.period, a)
            for m_miss in missing_list:
                if m_miss in orders:
                    ang_miss_d = math.degrees(orders[m_miss])
                    if -48.0 <= ang_miss_d <= 48.0:
                        missing_pts.append([ang_miss_d, 1e-4 if self.log_scale else 0.0])
                        missing_labels.append(m_miss)

        if missing_pts:
            self.scatter_missing.set_offsets(np.array(missing_pts))
            self.scatter_missing.set_visible(True)
        else:
            self.scatter_missing.set_offsets(np.empty((0, 2)))
            self.scatter_missing.set_visible(False)

        for i, txt in enumerate(self.txt_missing_pool):
            if i < len(missing_pts):
                ang_miss_d, _ = missing_pts[i]
                m_miss = missing_labels[i]
                txt.set_position((ang_miss_d, 2e-4 if self.log_scale else 0.04))
                txt.set_text(f"m={m_miss:+} (0)")
                txt.set_visible(True)
            else:
                txt.set_visible(False)

        # ─── 3. Detector Array: Simulated Screen Pattern ───
        screen_y_cm = propagation.sin_theta_to_screen_position(np.radians(theta_deg), self.screen_dist) * 100.0
        y_screen_linear = np.linspace(-30.0, 30.0, 400)
        i_screen = np.interp(y_screen_linear, screen_y_cm, intensity, left=0.0, right=0.0)

        # Realistic Gaussian vertical cross-section for laser spots
        y_vert = np.linspace(-1.0, 1.0, 25)[:, np.newaxis]
        strip_2d = np.exp(-y_vert**2 / 0.35) * i_screen[np.newaxis, :]

        laser_cmap = get_laser_colormap(self.s_wl.val)
        self.im_screen.set_data(strip_2d)
        self.im_screen.set_cmap(laser_cmap)

        screen_pts = []
        idx_s = 0
        for m, ang_rad in order_items:
            y_m = self.screen_dist * math.tan(ang_rad) * 100.0
            if -28.0 <= y_m <= 28.0:
                screen_pts.append([y_m, 1.35])
                if idx_s < len(self.txt_screen_pool):
                    txt = self.txt_screen_pool[idx_s]
                    txt.set_position((y_m, 1.48))
                    txt.set_text(f"m={m:+d}" if m != 0 else "m=0")
                    txt.set_visible(True)
                    idx_s += 1

        if screen_pts:
            self.scatter_screen_pointers.set_offsets(np.array(screen_pts))
            self.scatter_screen_pointers.set_visible(True)
        else:
            self.scatter_screen_pointers.set_offsets(np.empty((0, 2)))
            self.scatter_screen_pointers.set_visible(False)

        for j in range(idx_s, len(self.txt_screen_pool)):
            self.txt_screen_pool[j].set_visible(False)

        self.title_screen.set_text(
            f"DETECTOR ARRAY // SIMULATED SCREEN PATTERN (L = {self.screen_dist:.2f} m  |  λ = {int(self.s_wl.val)} nm  |  y = L·tan θ)"
        )

        # ─── 4. Bloomberg Telemetry Terminal Feed ───
        z_t_um = (2.0 * self.period**2 / self.wavelength) * 1e6
        telemetry_lines = [
            f"SYS: λ={self.wavelength*1e9:.0f}nm  d={self.period*1e6:.2f}µm  N={self.n_slits}  n={self.n_index:.2f}",
            f"FREQ: {lines_per_mm:.0f} lines/mm  |  z_T = {z_t_um:.2f} µm  |  {mat_name.split()[0]}",
            "─" * 40,
            "ORD   DIFF-ANGLE    DET-OFFSET   DISPERSION",
        ]
        for m in sorted(orders.keys()):
            ang_deg = math.degrees(orders[m])
            y_cm = self.screen_dist * math.tan(orders[m]) * 100.0
            cos_th = math.cos(orders[m])
            disp = (abs(m) / (self.period * cos_th)) * (180.0 / math.pi) * 1e-9 if cos_th > 1e-4 else 0.0
            blaze_flag = "  ◀ BLAZE" if (self.grating_type == "Blazed Phase" and m == 1) else ""
            telemetry_lines.append(f"m={m:+2d}   {ang_deg:+7.2f}°    {y_cm:+7.1f} cm    {disp:0.3f}°/nm{blaze_flag}")

        telemetry_lines.append("─" * 40)
        if self.grating_type == "Binary Amplitude":
            missing = analytical.missing_orders(self.period, a)
            if missing:
                missing_str = ", ".join(f"±{x}" for x in missing[:4])
                telemetry_lines.append(f"MISSING ORDERS : m = {missing_str} ({int(self.duty*100)}% duty)")
            else:
                telemetry_lines.append("MISSING ORDERS : None detected")
        elif self.grating_type == "Blazed Phase":
            telemetry_lines.append(f"PRIMARY BLAZE  : m = +1 (Efficiency ~95%) | h = {h_relief_nm:.1f} nm")
        elif self.grating_type == "Sinusoidal Phase":
            telemetry_lines.append(f"PHASE DEPTH    : φ₀ = {self.phase_depth:.2f} rad (h = {h_relief_nm:.1f} nm)")
        else:
            telemetry_lines.append("2D PROFILE     : 2-axis orthogonal grid")

        if evanescent:
            telemetry_lines.append(f"EVANESCENT     : |m| ≥ {min(abs(e) for e in evanescent)} (sin θ > 1)")
        else:
            telemetry_lines.append("EVANESCENT     : All modeled orders propagate")

        self.term_text.set_text("\n".join(telemetry_lines))

        if fast_mode and not full_draw:
            did_blit = self._fast_draw()
            if not did_blit:
                self.fig.canvas.draw_idle()
        else:
            self.fig.canvas.draw_idle()

    def open_talbot_window(self, _event=None) -> None:
        """Launch the near-field Talbot carpet simulation in a dark research-grade window."""
        fig_t = plt.figure(figsize=(10, 6.5), dpi=100, facecolor="#0b0f14")
        if hasattr(fig_t.canvas, "manager") and fig_t.canvas.manager is not None:
            fig_t.canvas.manager.set_window_title("GratingLab // Near-Field Talbot Carpet Analyzer")
        ax_t = fig_t.add_subplot(111, facecolor="#11161d")

        z_t = analytical.talbot_distance(self.period, self.wavelength)
        grid = gratings.Grid1D(width=20.0 * self.period, num_samples=1024)
        t_grating = gratings.binary_amplitude(grid.x, d=self.period, a=self.duty * self.period)

        x, z_arr, carpet = propagation.talbot_carpet(
            field=t_grating,
            dx=grid.dx,
            wavelength=self.wavelength,
            d=self.period,
            n_z_steps=180,
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
    n_index: float = 1.50,
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
        n_index=n_index,
    )
    plt.show()
