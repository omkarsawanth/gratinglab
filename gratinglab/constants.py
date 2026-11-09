"""
gratinglab.constants
====================
Physical constants, reference laser wavelengths, optical disc track pitches,
and unit conversion helper functions.

All internal calculations in GratingLab strictly follow SI units:
- Length: meters (m)
- Angles: radians (rad)
- Frequencies: Hertz (Hz)
"""

from __future__ import annotations
import math

# ---------------------------------------------------------------------------
# Fundamental Physical Constants (CODATA 2018 / SI definitions)
# ---------------------------------------------------------------------------
SPEED_OF_LIGHT: float = 299_792_458.0  # Vacuum speed of light [m/s]
PLANCK_CONSTANT: float = 6.626_070_15e-34  # Planck constant [J*s]

# ---------------------------------------------------------------------------
# Common Laser Wavelengths [meters]
# ---------------------------------------------------------------------------
WAVELENGTH_VIOLET_405: float = 405.0e-9  # GaN diode laser (405 nm)
WAVELENGTH_ARGON_488: float = 488.0e-9   # Argon-ion laser line (488 nm)
WAVELENGTH_GREEN_532: float = 532.0e-9   # Frequency-doubled Nd:YAG / DPSS (532 nm)
WAVELENGTH_HENE_632_8: float = 632.8e-9  # Helium-Neon red laser (632.8 nm)
WAVELENGTH_RED_650: float = 650.0e-9     # Standard red laser diode / DVD pointer (650 nm)
WAVELENGTH_NIR_780: float = 780.0e-9     # Near-infrared laser diode / CD pickup (780 nm)

# Lookup dictionary for convenience
LASER_SOURCES: dict[str, float] = {
    "violet_405nm": WAVELENGTH_VIOLET_405,
    "argon_488nm": WAVELENGTH_ARGON_488,
    "green_532nm": WAVELENGTH_GREEN_532,
    "hene_632.8nm": WAVELENGTH_HENE_632_8,
    "red_650nm": WAVELENGTH_RED_650,
    "nir_780nm": WAVELENGTH_NIR_780,
}

# ---------------------------------------------------------------------------
# Optical Media Track Pitches (Grating Spacings) [meters]
# ---------------------------------------------------------------------------
# Standard CD track pitch: 1.6 micrometers
PITCH_CD: float = 1.6e-6

# Standard DVD track pitch: 0.74 micrometers
PITCH_DVD: float = 0.74e-6

# Standard Blu-ray Disc (BD) track pitch: 0.32 micrometers
PITCH_BLURAY: float = 0.32e-6

OPTICAL_MEDIA_PITCHES: dict[str, float] = {
    "CD": PITCH_CD,
    "DVD": PITCH_DVD,
    "Blu-ray": PITCH_BLURAY,
}

# ---------------------------------------------------------------------------
# Common Laboratory Transmission Grating Ruling Densities
# ---------------------------------------------------------------------------
RULING_100_L_PER_MM: float = 1.0 / (100.0 * 1e3)   # 10.0 um pitch
RULING_300_L_PER_MM: float = 1.0 / (300.0 * 1e3)   # ~3.333 um pitch
RULING_600_L_PER_MM: float = 1.0 / (600.0 * 1e3)   # ~1.667 um pitch
RULING_1200_L_PER_MM: float = 1.0 / (1200.0 * 1e3) # ~0.833 um pitch

# ---------------------------------------------------------------------------
# Unit Conversion Utilities
# ---------------------------------------------------------------------------
def nm_to_m(val_nm: float) -> float:
    """Convert nanometers (nm) to meters (m)."""
    return val_nm * 1e-9


def m_to_nm(val_m: float) -> float:
    """Convert meters (m) to nanometers (nm)."""
    return val_m * 1e9


def um_to_m(val_um: float) -> float:
    """Convert micrometers (um) to meters (m)."""
    return val_um * 1e-6


def m_to_um(val_m: float) -> float:
    """Convert meters (m) to micrometers (um)."""
    return val_m * 1e6


def mm_to_m(val_mm: float) -> float:
    """Convert millimeters (mm) to meters (m)."""
    return val_mm * 1e-3


def m_to_mm(val_m: float) -> float:
    """Convert meters (m) to millimeters (mm)."""
    return val_m * 1e3


def deg_to_rad(deg: float) -> float:
    """Convert degrees to radians."""
    return math.radians(deg)


def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees."""
    return math.degrees(rad)


def lines_per_mm_to_pitch(lines_per_mm: float) -> float:
    """
    Convert ruling density in lines/mm to grating pitch (d) in meters.
    Equation: d = 1 / (N * 10^3)
    """
    if lines_per_mm <= 0:
        raise ValueError("Ruling density must be positive.")
    return 1.0 / (lines_per_mm * 1e3)


def pitch_to_lines_per_mm(pitch_m: float) -> float:
    """
    Convert grating pitch (d) in meters to ruling density in lines/mm.
    Equation: N = 1 / (d * 10^3)
    """
    if pitch_m <= 0:
        raise ValueError("Pitch must be positive.")
    return 1.0 / (pitch_m * 1e3)
