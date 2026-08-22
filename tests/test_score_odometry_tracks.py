from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/score_odometry_tracks.py"


def load_module():
    spec = importlib.util.spec_from_file_location("score_odometry_tracks", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quaternion_z(yaw: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.zeros(len(yaw)), np.zeros(len(yaw)), np.sin(yaw / 2), np.cos(yaw / 2)]
    )


def test_relative_scorer_detects_pelvis_frame_improvement() -> None:
    module = load_module()
    query_t = np.arange(0.0, 4.0, 0.02)
    motive = np.column_stack([0.1 * query_t, 0.02 * np.sin(query_t)])
    motive_yaw = 0.05 * query_t
    pelvis_position = np.column_stack([motive, np.zeros(len(query_t))])
    sensor_position = pelvis_position.copy()
    sensor_position[:, 0] += 0.08 * np.sin(3.0 * query_t)
    pelvis_quaternion = quaternion_z(motive_yaw)
    sensor_quaternion = quaternion_z(motive_yaw + 0.12 * np.sin(3.0 * query_t))

    pelvis, _ = module.score_track(
        query_t=query_t,
        motive_planar=motive,
        motive_yaw=motive_yaw,
        track_t=query_t,
        track_position=pelvis_position,
        track_quaternion=pelvis_quaternion,
    )
    sensor, _ = module.score_track(
        query_t=query_t,
        motive_planar=motive,
        motive_yaw=motive_yaw,
        track_t=query_t,
        track_position=sensor_position,
        track_quaternion=sensor_quaternion,
    )

    assert pelvis["planar_position_error_m"]["rmse"] < 1e-9
    assert pelvis["relative_yaw_error_deg"]["rmse"] < 1e-9
    assert sensor["planar_position_error_m"]["rmse"] > 0.03
    assert sensor["relative_yaw_error_deg"]["rmse"] > 3.0


def test_yaw_extractors_match_expected_up_axes() -> None:
    module = load_module()
    yaw = np.array([0.0, 0.2, 0.4])
    qz = quaternion_z(yaw)
    qy = np.column_stack(
        [np.zeros(len(yaw)), np.sin(yaw / 2), np.zeros(len(yaw)), np.cos(yaw / 2)]
    )
    np.testing.assert_allclose(module.yaw_z_up(qz), yaw)
    np.testing.assert_allclose(module.yaw_y_up(qy), yaw)
