"""Tests for gratinglab.constants"""

import math
import pytest
from gratinglab import constants


def test_wavelength_values():
    assert constants.WAVELENGTH_RED_650 == 650.0e-9
    assert constants.WAVELENGTH_GREEN_532 == 532.0e-9
    assert constants.WAVELENGTH_HENE_632_8 == pytest.approx(632.8e-9)


def test_optical_media_pitches():
    assert constants.PITCH_CD == 1.6e-6
    assert constants.PITCH_DVD == 0.74e-6
    assert constants.PITCH_BLURAY == 0.32e-6


def test_unit_conversions():
    assert constants.nm_to_m(650.0) == 650.0e-9
    assert constants.m_to_nm(650.0e-9) == pytest.approx(650.0)
    assert constants.um_to_m(1.6) == 1.6e-6
    assert constants.m_to_um(1.6e-6) == pytest.approx(1.6)
    assert constants.deg_to_rad(180.0) == pytest.approx(math.pi)
    assert constants.rad_to_deg(math.pi) == pytest.approx(180.0)


def test_ruling_density_conversion():
    pitch_600 = constants.lines_per_mm_to_pitch(600.0)
    assert pitch_600 == pytest.approx(1.0 / (600.0 * 1e3))
    assert constants.pitch_to_lines_per_mm(pitch_600) == pytest.approx(600.0)
