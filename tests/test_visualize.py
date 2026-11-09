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
