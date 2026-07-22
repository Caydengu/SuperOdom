from __future__ import annotations

import math

import pytest

from g1_root_state_bridge.joint_contract import (
    CANONICAL_G1_JOINT_NAMES,
    JointContractError,
    TimedJointSample,
)


def sample_in_order(names: tuple[str, ...]) -> TimedJointSample:
    values = {name: index + 0.25 for index, name in enumerate(CANONICAL_G1_JOINT_NAMES)}
    velocities = {name: -(index + 0.5) for index, name in enumerate(CANONICAL_G1_JOINT_NAMES)}
    return TimedJointSample(
        stamp_ns=1_000_000_000,
        receipt_ns=1_002_000_000,
        names=names,
        position=tuple(values[name] for name in names),
        velocity=tuple(velocities[name] for name in names),
        sequence=123,
    )


def test_canonicalization_is_name_based_and_preserves_asymmetric_waist_values() -> None:
    unordered = sample_in_order(tuple(reversed(CANONICAL_G1_JOINT_NAMES)))
    canonical = unordered.canonicalized()

    assert canonical.names == CANONICAL_G1_JOINT_NAMES
    assert canonical.position[0] == 0.25
    assert canonical.position[6] == 6.25
    assert canonical.position[12:15] == (12.25, 13.25, 14.25)
    assert canonical.position[15] == 15.25
    assert canonical.position[22] == 22.25
    assert canonical.velocity[12:15] == (-12.5, -13.5, -14.5)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"names": CANONICAL_G1_JOINT_NAMES[:-1]}, "exactly 29"),
        (
            {"names": CANONICAL_G1_JOINT_NAMES[:-1] + (CANONICAL_G1_JOINT_NAMES[0],)},
            "duplicates",
        ),
        ({"position": (math.nan,) + (0.0,) * 28}, "finite"),
        ({"velocity": (0.0,) * 28}, "length"),
        ({"stamp_ns": -1}, "stamp_ns"),
        ({"receipt_ns": 999_999_999}, "receipt_ns"),
        ({"sequence": -1}, "sequence"),
        ({"source_epoch": -1}, "source_epoch"),
    ],
)
def test_joint_contract_rejects_invalid_samples(mutation: dict[str, object], match: str) -> None:
    values: dict[str, object] = {
        "stamp_ns": 1_000_000_000,
        "receipt_ns": 1_002_000_000,
        "names": CANONICAL_G1_JOINT_NAMES,
        "position": (0.0,) * 29,
        "velocity": (0.0,) * 29,
        "sequence": 1,
        "source_epoch": 7,
    }
    values.update(mutation)
    with pytest.raises(JointContractError, match=match):
        TimedJointSample(**values)


def test_canonical_names_match_unitree_motor_order() -> None:
    assert CANONICAL_G1_JOINT_NAMES == (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
