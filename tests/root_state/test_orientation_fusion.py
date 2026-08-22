from __future__ import annotations

import math

import numpy as np
import pytest
from g1_root_state_bridge.orientation_fusion import (
    OrientationFusionConfig,
    PelvisOrientationFusion,
    gravity_aligned_heading,
)
from g1_root_state_bridge.root_imu_buffer import RootImuBuffer
from g1_root_state_bridge.root_imu_contract import TimedRootImuSample


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def sample(stamp_ns: int, sequence: int, *, gyro_z: float = 0.0) -> TimedRootImuSample:
    return TimedRootImuSample(
        stamp_ns=stamp_ns,
        receipt_ns=stamp_ns + 1_000_000,
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, gyro_z),
        linear_acceleration=(0.0, 0.0, 9.81),
        sequence=sequence,
        source_epoch=9,
    )


def yaw(rotation: np.ndarray) -> float:
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def test_root_imu_buffer_interpolates_at_estimator_source_time() -> None:
    buffer = RootImuBuffer()
    buffer.append(sample(1_000_000_000, 1, gyro_z=0.0))
    buffer.append(sample(1_010_000_000, 2, gyro_z=1.0))
    match = buffer.synchronize(1_004_000_000, max_gap_ns=10_000_000)
    assert match.sample.stamp_ns == 1_004_000_000
    assert match.sample.angular_velocity[2] == 0.4
    assert match.maximum_bracket_gap_ns == 6_000_000


def test_fusion_uses_root_gyro_between_lio_yaw_anchors() -> None:
    fusion = PelvisOrientationFusion(
        OrientationFusionConfig(yaw_anchor_time_constant_s=10.0)
    )
    first = fusion.update(
        source_time_ns=1_000_000_000,
        local_world_R_pelvis=rot_z(0.0),
        root_imu=sample(1_000_000_000, 1, gyro_z=1.0),
    )
    second = fusion.update(
        source_time_ns=1_010_000_000,
        local_world_R_pelvis=rot_z(0.0),
        root_imu=sample(1_010_000_000, 2, gyro_z=1.0),
    )
    assert first.healthy and second.healthy
    assert 0.009 < yaw(second.world_R_pelvis) < 0.011


def test_fusion_fails_closed_on_large_yaw_innovation() -> None:
    fusion = PelvisOrientationFusion(
        OrientationFusionConfig(maximum_yaw_innovation_rad=0.2)
    )
    fusion.update(
        source_time_ns=1_000_000_000,
        local_world_R_pelvis=rot_z(0.0),
        root_imu=sample(1_000_000_000, 1),
    )
    result = fusion.update(
        source_time_ns=1_010_000_000,
        local_world_R_pelvis=rot_z(1.0),
        root_imu=sample(1_010_000_000, 2),
    )
    assert not result.healthy
    assert result.reason == "yaw_innovation"
    assert np.allclose(result.world_R_pelvis, rot_z(1.0))


def test_gravity_aligned_heading_preserves_lio_yaw_without_state() -> None:
    half_roll = math.radians(12.0) / 2.0
    result = gravity_aligned_heading(
        heading_rotation=rot_z(math.radians(31.0)),
        gravity_quaternion_wxyz=(math.cos(half_roll), math.sin(half_roll), 0.0, 0.0),
    )
    assert math.degrees(yaw(result)) == pytest.approx(31.0)
    assert math.degrees(math.atan2(result[2, 1], result[2, 2])) == pytest.approx(12.0)
