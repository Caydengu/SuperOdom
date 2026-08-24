from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.score_robot_vlm_map_localization import (
    motive_pelvis_reference_in_map,
    score_map_trace,
)


def test_motive_front_plane_offset_and_yaw_enter_canonical_map() -> None:
    theta = 0.4
    quaternion = np.asarray(
        ((0.0, math.sin(theta / 2), 0.0, math.cos(theta / 2)),)
    )
    position = np.asarray(((1.0, 0.8, 2.0),))
    T_map_motive = np.asarray(
        (
            (1.0, 0.0, 0.0, 3.0),
            (0.0, 0.0, -1.0, 4.0),
            (0.0, 1.0, 0.0, -0.8),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    mapped, yaw = motive_pelvis_reference_in_map(
        position,
        quaternion,
        T_map_motive=T_map_motive,
        front_plane_to_pelvis_x_m=-0.06,
        pelvis_yaw_offset_rad=0.1,
    )
    assert mapped.shape == (1, 3)
    assert yaw[0] == pytest.approx(theta + 0.1)
    assert mapped[0, 2] == pytest.approx(0.0)
    front_plane_map = T_map_motive @ np.r_[position[0], 1.0]
    assert mapped[0, 0] == pytest.approx(front_plane_map[0] - 0.06 * math.cos(theta + 0.1))
    assert mapped[0, 1] == pytest.approx(front_plane_map[1] - 0.06 * math.sin(theta + 0.1))


def test_absolute_map_score_does_not_align_away_global_error() -> None:
    reference_time = np.asarray((1_000_000_000, 2_000_000_000, 3_000_000_000))
    reference_position = np.asarray(((0.0, 0.0, 0.8), (1.0, 0.0, 0.8), (2.0, 0.0, 0.8)))
    reference_yaw = np.zeros(3)
    trace = [
        {
            "receipt_realtime_ns": int(time_ns + 10_000_000),
            "local_estimate_realtime_ns": int(time_ns),
            "base_pose_xyyaw": [float(index) + 0.2, 0.0, 0.1],
        }
        for index, time_ns in enumerate(reference_time)
    ]
    metrics, enriched = score_map_trace(
        trace,
        reference_time_ns=reference_time,
        reference_position_xyz_m=reference_position,
        reference_yaw_rad=reference_yaw,
        T_map_px=np.eye(3),
    )
    assert metrics["posthoc_alignment_applied"] is False
    assert metrics["absolute_planar_error_m"]["rmse"] == pytest.approx(0.2)
    assert metrics["absolute_yaw_error_deg"]["rmse"] == pytest.approx(
        math.degrees(0.1)
    )
    assert len(enriched) == 3
