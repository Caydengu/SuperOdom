from __future__ import annotations

import math

import numpy as np
import pytest

from g1_root_state_bridge.kiss_pelvis_local_odometry import (
    KissPelvisContractError,
    KissPelvisObservation,
    KissPelvisPacketBuilder,
)
from g1_root_state_bridge.protocol import (
    REQUIRED_ROOT_FUSION_FLAGS,
    RootStateHealth,
    serialize_root_state_v2,
)


def pose(x: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    value = np.eye(4)
    value[:3, :3] = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    value[:3, 3] = (x, 0.0, 0.8)
    return value


def observation(time_ns: int, value: np.ndarray, **changes: object) -> KissPelvisObservation:
    base = dict(
        source_epoch=7,
        estimate_time_ns=time_ns,
        correction_time_ns=time_ns - 50_000_000,
        joint_time_ns=time_ns + 1_000_000,
        publish_time_ns=time_ns + 2_000_000,
        receipt_time_ns=time_ns + 3_000_000,
        local_T_pelvis=value,
        covariance_diagonal=(0.01,) * 6,
        registration_valid=True,
        deskew_valid=True,
        heading_valid=True,
        calibration_valid=True,
        clock_valid=True,
    )
    base.update(changes)
    return KissPelvisObservation(**base)


def test_builder_emits_fused_health_and_pose_derived_twist() -> None:
    builder = KissPelvisPacketBuilder(calibration_digest=b"c" * 32)
    first = builder.build(observation(1_000_000_000, pose(0.0, 0.0)))
    second = builder.build(observation(1_100_000_000, pose(0.1, 0.02)))
    assert first.sequence == 1
    assert second.sequence == 2
    assert (second.health_flags & REQUIRED_ROOT_FUSION_FLAGS) == REQUIRED_ROOT_FUSION_FLAGS
    assert second.linear_velocity == pytest.approx((1.0, 0.0, 0.0))
    assert second.angular_velocity == pytest.approx((0.0, 0.0, 0.2))
    assert len(serialize_root_state_v2(second)) == 176


def test_missing_deskew_or_heading_evidence_stays_explicit() -> None:
    builder = KissPelvisPacketBuilder(calibration_digest=b"c" * 32)
    packet = builder.build(
        observation(
            1_000_000_000,
            pose(0.0, 0.0),
            deskew_valid=False,
            heading_valid=False,
        )
    )
    assert not packet.health_flags & RootStateHealth.INERTIAL_DESKEW_VALID
    assert not packet.health_flags & RootStateHealth.HEADING_VALID
    assert packet.strictly_valid


def test_epoch_reset_restarts_sequence_but_time_regression_within_epoch_fails() -> None:
    builder = KissPelvisPacketBuilder(calibration_digest=b"c" * 32)
    builder.build(observation(1_000_000_000, pose(0.0, 0.0)))
    with pytest.raises(KissPelvisContractError, match="estimate time"):
        builder.build(observation(900_000_000, pose(0.0, 0.0)))
    reset = builder.build(
        observation(900_000_000, pose(0.0, 0.0), source_epoch=8)
    )
    assert reset.sequence == 1
    assert reset.source_epoch == 8
