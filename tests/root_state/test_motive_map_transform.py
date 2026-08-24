from __future__ import annotations

import copy

import numpy as np

from scripts.fit_motive_map_transform import fit_control_points


def _fixture(noise=0.0):
    angle = np.deg2rad(27.0)
    # determinant -1: the only branch consistent with positive Motive Y-up and map Z-up.
    A = np.asarray(((np.cos(angle), np.sin(angle)), (np.sin(angle), -np.cos(angle))))
    t = np.asarray((1.2, -0.8))
    motive = np.asarray(
        ((0.0, 0.02, 0.0), (2.0, 0.01, 0.0), (0.0, 0.00, 2.0),
         (2.5, 0.03, 1.5), (-1.0, 0.02, 2.4), (1.2, 0.01, -1.4))
    )
    map_xyz = np.column_stack(((A @ motive[:, (0, 2)].T).T + t, motive[:, 1] + 0.3))
    if noise:
        map_xyz += np.random.default_rng(4).normal(0.0, noise, map_xyz.shape)
    return {
        "schema": "g1_motive_polycam_control_points_v1",
        "map_identity": {
            "glb_sha256": "a" * 64,
            "surface_sha256": "b" * 64,
            "structural_map_sha256": "c" * 64,
        },
        "motive_contract": {"up_axis": "y", "horizontal_axes": ["x", "z"]},
        "points": [
            {
                "label": f"p{index}",
                "role": "fit" if index < 5 else "holdout",
                "motive_xyz_m": source.tolist(),
                "map_xyz_m": target.tolist(),
            }
            for index, (source, target) in enumerate(zip(motive, map_xyz))
        ],
    }


def test_fit_motive_map_transform_recovers_metric_right_handed_mapping():
    result = fit_control_points(_fixture(noise=0.003))
    assert result["status"] == "accepted"
    assert np.isclose(result["metrics"]["horizontal_determinant"], -1.0)
    assert np.isclose(result["metrics"]["rotation_determinant"], 1.0)
    assert np.isclose(result["metrics"]["up_alignment_dot"], 1.0)
    assert result["metrics"]["holdout_max_m"] < 0.02


def test_fit_rejects_a_bad_held_out_landmark():
    value = _fixture()
    value["points"][-1]["map_xyz_m"][0] += 0.25
    result = fit_control_points(value)
    assert result["status"] == "rejected"
    assert "holdout" in result["failures"]


def test_fit_rejects_wrong_axis_contract_and_requires_holdout():
    value = _fixture()
    wrong = copy.deepcopy(value)
    wrong["motive_contract"]["up_axis"] = "z"
    try:
        fit_control_points(wrong)
    except ValueError as error:
        assert "Y-up" in str(error)
    else:
        raise AssertionError("wrong Motive axis contract was accepted")
    value["points"][-1]["role"] = "fit"
    try:
        fit_control_points(value)
    except ValueError as error:
        assert "held-out" in str(error)
    else:
        raise AssertionError("calibration without a holdout was accepted")
