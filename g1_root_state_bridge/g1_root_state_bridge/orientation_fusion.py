"""Source-time pelvis orientation fusion anchored to local LiDAR odometry."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from g1_root_state_bridge.root_imu_contract import TimedRootImuSample


def _shortest(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _rpy_from_rotation(rotation: np.ndarray) -> tuple[float, float, float]:
    pitch = math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return roll, pitch, yaw


def _rotation_from_wxyz(quaternion: tuple[float, ...]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class OrientationFusionConfig:
    yaw_anchor_time_constant_s: float = 0.75
    maximum_step_s: float = 0.05
    maximum_yaw_innovation_rad: float = math.radians(45.0)
    gyro_z_sign: float = 1.0

    def __post_init__(self) -> None:
        if self.yaw_anchor_time_constant_s <= 0.0:
            raise ValueError("yaw anchor time constant must be positive")
        if self.maximum_step_s <= 0.0:
            raise ValueError("maximum fusion step must be positive")
        if not 0.0 < self.maximum_yaw_innovation_rad <= math.pi:
            raise ValueError("maximum yaw innovation must be in (0, pi]")
        if self.gyro_z_sign not in (-1.0, 1.0):
            raise ValueError("gyro_z_sign must be exactly -1 or +1")


@dataclass(frozen=True)
class OrientationFusionResult:
    world_R_pelvis: np.ndarray
    angular_velocity_world: np.ndarray
    healthy: bool
    reason: str
    source_time_ns: int
    yaw_innovation_rad: float


class PelvisOrientationFusion:
    """Use root-IMU gravity and yaw rate while retaining LIO yaw as anchor.

    The root IMU is not inserted into the head-sensor preintegrator.  Instead,
    synchronized waist FK first recovers the pelvis pose; this filter then
    replaces pelvis roll/pitch with root gravity and propagates yaw with the
    pelvis gyro between source-timestamped LIO observations.
    """

    def __init__(self, config: OrientationFusionConfig = OrientationFusionConfig()):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._last_time_ns: int | None = None
        self._last_source_epoch: int | None = None
        self._fused_yaw = 0.0

    def update(
        self,
        *,
        source_time_ns: int,
        local_world_R_pelvis: np.ndarray,
        root_imu: TimedRootImuSample,
    ) -> OrientationFusionResult:
        local_rotation = np.asarray(local_world_R_pelvis, dtype=np.float64)
        if local_rotation.shape != (3, 3) or not np.all(np.isfinite(local_rotation)):
            raise ValueError("local_world_R_pelvis must be one finite rotation")
        imu_rotation = _rotation_from_wxyz(root_imu.quaternion_wxyz)
        roll, pitch, _imu_yaw = _rpy_from_rotation(imu_rotation)
        _local_roll, _local_pitch, local_yaw = _rpy_from_rotation(local_rotation)

        initialize = (
            self._last_time_ns is None
            or self._last_source_epoch != root_imu.source_epoch
        )
        if initialize:
            self._last_time_ns = source_time_ns
            self._last_source_epoch = root_imu.source_epoch
            self._fused_yaw = local_yaw
            fused = _rotation_from_rpy(roll, pitch, self._fused_yaw)
            return OrientationFusionResult(
                world_R_pelvis=fused,
                angular_velocity_world=fused @ np.asarray(root_imu.angular_velocity),
                healthy=True,
                reason="initialized",
                source_time_ns=source_time_ns,
                yaw_innovation_rad=0.0,
            )

        assert self._last_time_ns is not None
        dt = (source_time_ns - self._last_time_ns) * 1e-9
        self._last_time_ns = source_time_ns
        if dt <= 0.0:
            return OrientationFusionResult(
                world_R_pelvis=local_rotation.copy(),
                angular_velocity_world=local_rotation @ np.asarray(root_imu.angular_velocity),
                healthy=False,
                reason="non_monotonic_source_time",
                source_time_ns=source_time_ns,
                yaw_innovation_rad=0.0,
            )
        if dt > self.config.maximum_step_s:
            self._fused_yaw = local_yaw
            return OrientationFusionResult(
                world_R_pelvis=local_rotation.copy(),
                angular_velocity_world=local_rotation @ np.asarray(root_imu.angular_velocity),
                healthy=False,
                reason="source_gap",
                source_time_ns=source_time_ns,
                yaw_innovation_rad=0.0,
            )

        predicted = _shortest(
            self._fused_yaw
            + self.config.gyro_z_sign * root_imu.angular_velocity[2] * dt
        )
        innovation = _shortest(local_yaw - predicted)
        if abs(innovation) > self.config.maximum_yaw_innovation_rad:
            self._fused_yaw = local_yaw
            return OrientationFusionResult(
                world_R_pelvis=local_rotation.copy(),
                angular_velocity_world=local_rotation @ np.asarray(root_imu.angular_velocity),
                healthy=False,
                reason="yaw_innovation",
                source_time_ns=source_time_ns,
                yaw_innovation_rad=innovation,
            )
        gain = 1.0 - math.exp(-dt / self.config.yaw_anchor_time_constant_s)
        self._fused_yaw = _shortest(predicted + gain * innovation)
        fused = _rotation_from_rpy(roll, pitch, self._fused_yaw)
        return OrientationFusionResult(
            world_R_pelvis=fused,
            angular_velocity_world=fused @ np.asarray(root_imu.angular_velocity),
            healthy=True,
            reason="fused",
            source_time_ns=source_time_ns,
            yaw_innovation_rad=innovation,
        )
