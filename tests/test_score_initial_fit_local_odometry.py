from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.score_initial_fit_local_odometry import _load_odometry


def test_load_odometry_selects_topic_and_preserves_pose(tmp_path: Path) -> None:
    path = tmp_path / "odometry.jsonl"
    rows = [
        {"kind": "metadata", "topics": ["/g1/localization/pelvis_odom"]},
        {
            "kind": "odometry",
            "topic": "/other",
            "source_time_ns": 9,
            "position_xyz_m": [9.0, 9.0, 9.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        {
            "kind": "odometry",
            "topic": "/g1/localization/pelvis_odom",
            "source_time_ns": 10,
            "position_xyz_m": [1.0, 2.0, 3.0],
            "quaternion_xyzw": [0.0, 0.0, 0.1, 0.99],
        },
        {
            "kind": "odometry",
            "topic": "/g1/localization/pelvis_odom",
            "source_time_ns": 20,
            "position_xyz_m": [4.0, 5.0, 6.0],
            "quaternion_xyzw": [0.0, 0.0, 0.2, 0.98],
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert _load_odometry(path, "/g1/localization/pelvis_odom") == [
        {
            "source_time_ns": 10,
            "position_xyz_m": [1.0, 2.0, 3.0],
            "quaternion_xyzw": [0.0, 0.0, 0.1, 0.99],
        },
        {
            "source_time_ns": 20,
            "position_xyz_m": [4.0, 5.0, 6.0],
            "quaternion_xyzw": [0.0, 0.0, 0.2, 0.98],
        },
    ]


def test_load_odometry_fails_closed_without_two_samples(tmp_path: Path) -> None:
    path = tmp_path / "odometry.jsonl"
    path.write_text(
        json.dumps(
            {
                "kind": "odometry",
                "topic": "/g1/localization/pelvis_odom",
                "source_time_ns": 10,
                "position_xyz_m": [1.0, 2.0, 3.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="missing odometry topic"):
        _load_odometry(path, "/g1/localization/pelvis_odom")
