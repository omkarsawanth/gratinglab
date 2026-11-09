"""Tests for gratinglab.case_studies"""

from gratinglab import case_studies


def test_study_strain_temperature_execution():
    res = case_studies.study_strain_temperature(num_steps=5, save_fig=False)
    assert len(res["applied_strains"]) == 5
    max_err = max(abs(res["recovered_strains"] - res["applied_strains"]))
    assert max_err < 2e-3


def test_study_groove_depth_execution():
    res = case_studies.study_groove_depth(unknown_depth_nm=480.0, save_fig=False)
    assert res["error_pct"] < 2.0


def test_study_2d_lattice_execution():
    res = case_studies.study_2d_lattice(save_fig=False)
    assert res["error_x_pct"] < 1.0
    assert res["error_y_pct"] < 1.0
