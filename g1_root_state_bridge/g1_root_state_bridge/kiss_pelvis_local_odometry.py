"""Fail-closed KISS-ICP pelvis samples for the shared ``HSROOT02`` lane.

The geometric estimator is deliberately outside this module: it consumes one
already deskewed, dynamic-FK pelvis pose at the physical LiDAR correction time.
This core owns the stateful contract that must remain identical in offline
replay and a future ROS publisher: monotonic epochs/sequences, finite poses,
source-time joint synchronization, health evidence, and pose-derived twist.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from g1_root_state_bridge.protocol import (
    RootStateHealth,
    RootStatePacketV2,
)


class KissPelvisContractError(ValueError):
    """One local-odometry observation violates the frozen producer contract."""


@dataclass(frozen=True)
class KissPelvisHealthConfig:
    maximum_joint_sync_gap_ns: int = 10_000_000
    maximum_correction_age_ns: int = 150_000_000

    def __post_init__(self) -> None:
        if self.maximum_joint_sync_gap_ns <= 0:
            raise KissPelvisContractError("joint sync limit must be positive")
        if self.maximum_correction_age_ns <= 0:
            raise KissPelvisContractError("correction age limit must be positive")


@dataclass(frozen=True)
class KissPelvisObservation:
    source_epoch: int
    estimate_time_ns: int
    correction_time_ns: int
    joint_time_ns: int
    publish_time_ns: int
    receipt_time_ns: int
    local_T_pelvis: np.ndarray
    covariance_diagonal: tuple[float, float, float, float, float, float]
    registration_valid: bool
    deskew_valid: bool
    heading_valid: bool
    calibration_valid: bool
    clock_valid: bool = True


def _validated_pose(value: np.ndarray) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise KissPelvisContractError("local_T_pelvis must be one finite 4x4 pose")
    if not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise KissPelvisContractError("local_T_pelvis must be homogeneous")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise KissPelvisContractError("local_T_pelvis rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise KissPelvisContractError("local_T_pelvis rotation must be proper")
    return pose.copy()


def _quaternion_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            values = (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            values = (
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            )
        else:
            scale = math.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            values = (
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            )
    quaternion = np.asarray(values, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    sine = math.sin(angle)
    if abs(sine) < 1e-8:
        # The walking data never approaches pi between 10 Hz scans.  Rejecting
        # this ambiguous numerical branch is safer than inventing an axis.
        raise KissPelvisContractError("inter-scan rotation is numerically ambiguous")
    axis = np.asarray(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ),
        dtype=np.float64,
    ) / (2.0 * sine)
    return axis * angle


class KissPelvisPacketBuilder:
    """Build monotonic local-lane packets without hiding missing evidence."""

    def __init__(
        self,
        *,
        calibration_digest: bytes,
        config: KissPelvisHealthConfig = KissPelvisHealthConfig(),
    ) -> None:
        if not isinstance(calibration_digest, bytes) or len(calibration_digest) != 32:
            raise KissPelvisContractError("calibration digest must contain 32 raw bytes")
        self.calibration_digest = calibration_digest
        self.config = config
        self._sequence = 0
        self._previous: tuple[int, np.ndarray] | None = None
        self._source_epoch: int | None = None

    def build(self, observation: KissPelvisObservation) -> RootStatePacketV2:
        pose = _validated_pose(observation.local_T_pelvis)
        if not isinstance(observation.source_epoch, int) or observation.source_epoch < 0:
            raise KissPelvisContractError("source epoch must be a non-negative integer")
        for name in (
            "estimate_time_ns",
            "correction_time_ns",
            "joint_time_ns",
            "publish_time_ns",
            "receipt_time_ns",
        ):
            value = getattr(observation, name)
            if not isinstance(value, int) or value < 0:
                raise KissPelvisContractError(f"{name} must be a non-negative integer")
        if len(observation.covariance_diagonal) != 6 or not all(
            math.isfinite(value) and value >= 0.0
            for value in observation.covariance_diagonal
        ):
            raise KissPelvisContractError(
                "covariance diagonal must contain six finite non-negative values"
            )
        ordered_clock = (
            observation.correction_time_ns <= observation.estimate_time_ns
            <= observation.publish_time_ns
            <= observation.receipt_time_ns
        )
        if not ordered_clock:
            raise KissPelvisContractError(
                "times must satisfy correction <= estimate <= publish <= receipt"
            )
        if self._source_epoch is not None and observation.source_epoch < self._source_epoch:
            raise KissPelvisContractError("source epoch regressed")
        if observation.source_epoch != self._source_epoch:
            self._source_epoch = observation.source_epoch
            self._previous = None
            self._sequence = 0
        if self._previous is not None and observation.estimate_time_ns <= self._previous[0]:
            raise KissPelvisContractError("estimate time did not increase")

        linear_velocity = np.zeros(3, dtype=np.float64)
        angular_velocity = np.zeros(3, dtype=np.float64)
        if self._previous is not None:
            previous_time_ns, previous_pose = self._previous
            dt_s = (observation.estimate_time_ns - previous_time_ns) * 1e-9
            linear_velocity = (pose[:3, 3] - previous_pose[:3, 3]) / dt_s
            relative_rotation = pose[:3, :3] @ previous_pose[:3, :3].T
            angular_velocity = _rotation_vector(relative_rotation) / dt_s

        flags = RootStateHealth(0)
        if observation.registration_valid:
            flags |= RootStateHealth.ESTIMATOR_HEALTHY
        flags |= RootStateHealth.FINITE_POSE
        joint_gap_ns = observation.joint_time_ns - observation.estimate_time_ns
        if abs(joint_gap_ns) <= self.config.maximum_joint_sync_gap_ns:
            flags |= RootStateHealth.JOINT_SYNC_VALID
        correction_age_ns = observation.estimate_time_ns - observation.correction_time_ns
        if 0 <= correction_age_ns <= self.config.maximum_correction_age_ns:
            flags |= RootStateHealth.CORRECTION_FRESH
        if observation.calibration_valid:
            flags |= RootStateHealth.CALIBRATION_VALID
        if observation.clock_valid:
            flags |= RootStateHealth.CLOCK_VALID
        if observation.deskew_valid:
            flags |= RootStateHealth.INERTIAL_DESKEW_VALID
        if observation.heading_valid:
            flags |= RootStateHealth.HEADING_VALID

        self._sequence += 1
        self._previous = (observation.estimate_time_ns, pose)
        return RootStatePacketV2(
            sequence=self._sequence,
            source_epoch=observation.source_epoch,
            estimate_time_ns=observation.estimate_time_ns,
            publish_time_ns=observation.publish_time_ns,
            correction_time_ns=observation.correction_time_ns,
            joint_time_ns=observation.joint_time_ns,
            joint_sync_gap_ns=joint_gap_ns,
            health_flags=flags,
            position=tuple(float(value) for value in pose[:3, 3]),
            quaternion_wxyz=_quaternion_wxyz(pose[:3, :3]),
            linear_velocity=tuple(float(value) for value in linear_velocity),
            angular_velocity=tuple(float(value) for value in angular_velocity),
            covariance_diagonal=observation.covariance_diagonal,
            calibration_digest=self.calibration_digest,
        )
