"""Tests for validation.cd_dvd_validation"""

import math
import os
import pytest
from validation import cd_dvd_validation


def test_analytical_pitch_inversion():
    L = 0.5
    wl = 650.0e-9
    d_true = 1.6e-6
    sin_theta = wl / d_true
    theta = math.asin(sin_theta)
    offset = L * math.tan(theta)

    records = [{
        "media": "CD",
        "screen_distance_m": L,
        "order_m": 1,
        "spot_offset_m": offset,
        "wavelength_nm": 650.0,
    }]

    processed = cd_dvd_validation.process_measurements(records)
    res = processed[0]

    assert pytest.approx(1.600, rel=1e-5) == res["d_meas_um"]
    assert pytest.approx(0.0, abs=1e-4) == res["err_pitch_pct"]
