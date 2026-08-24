from __future__ import annotations

import numpy as np

from scripts.analyze_live_odometry_stress import classify_motion


def test_classify_motion_is_reference_kinematics_only_and_order_independent() -> None:
    kinematics = {
        "speed_mps": np.asarray([0.01, 0.05, 0.30, 0.30, 0.30, 0.30]),
        "yaw_rate_radps": np.radians([0.0, 30.0, 2.0, 2.0, 2.0, 30.0]),
        "forward_speed_mps": np.asarray([0.0, 0.01, 0.29, 0.02, 0.20, 0.20]),
        "lateral_speed_mps": np.asarray([0.0, 0.01, 0.02, 0.29, 0.20, 0.20]),
    }

    assert classify_motion(kinematics).tolist() == [
        "stationary",
        "pure_turn",
        "longitudinal",
        "lateral",
        "diagonal",
        "translation_and_turn",
    ]


def test_classify_motion_keeps_low_speed_low_turn_transition_band() -> None:
    kinematics = {
        "speed_mps": np.asarray([0.05]),
        "yaw_rate_radps": np.radians([10.0]),
        "forward_speed_mps": np.asarray([0.04]),
        "lateral_speed_mps": np.asarray([0.01]),
    }
    assert classify_motion(kinematics).tolist() == ["transition"]
