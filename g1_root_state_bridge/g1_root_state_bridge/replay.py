"""Mechanics-only historical replay support for the pelvis bridge.

Historical SuperOdometry bags predate the typed calibration and G1 joint
sidecars.  This module deliberately labels the missing inputs as synthesized:
it can validate frame conversion, FK execution, packet continuity, and strict
transport semantics, but it cannot establish dynamic pelvis accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from g1_root_state_bridge.bridge_core import (
    BridgeCore,
    CorrectionSample,
    EstimatorCalibration,
    EstimatorHealthSample,
    EstimatorObservation,
    UNKNOWN_COVARIANCE_DIAGONAL,
)
from g1_root_state_bridge.joint_contract import (
    CANONICAL_G1_JOINT_NAMES,
    TimedJointSample,
)
from g1_root_state_bridge.kinematics import PelvisKinematics, _invert_transform
from g1_root_state_bridge.protocol import (
    RootStatePacketV2,
    serialize_root_state_v2,
)


def gravity_alignment_from_accelerations(
    accelerations: Iterable[np.ndarray],
    *,
    map_T_lidar: np.ndarray,
    imu_T_lidar: np.ndarray,
) -> np.ndarray:
    """Reconstruct the startup ``map_T_gravity`` convention from bagged IMU.

    This mirrors the estimator's typed-calibration construction: measured
    specific force defines up, while map +X projected onto the horizontal
    plane preserves yaw.
    """
    values = [np.asarray(value, dtype=np.float64).reshape(3) for value in accelerations]
    if not values:
        raise ValueError("gravity alignment requires acceleration samples")
    mean_acceleration_imu = np.mean(np.stack(values), axis=0)
    if not np.all(np.isfinite(mean_acceleration_imu)):
        raise ValueError("gravity acceleration samples must be finite")
    norm = float(np.linalg.norm(mean_acceleration_imu))
    if norm < 1e-6:
        raise ValueError("gravity direction is unobservable from acceleration")

    map_T_lidar = np.asarray(map_T_lidar, dtype=np.float64).reshape(4, 4)
    imu_T_lidar = np.asarray(imu_T_lidar, dtype=np.float64).reshape(4, 4)
    map_T_imu = map_T_lidar @ _invert_transform(imu_T_lidar)
    gravity_down_imu = -mean_acceleration_imu / norm
    gravity_down_map = map_T_imu[:3, :3] @ gravity_down_imu
    gravity_down_map /= np.linalg.norm(gravity_down_map)
    gravity_up_map = -gravity_down_map

    gravity_x_map = np.array((1.0, 0.0, 0.0), dtype=np.float64)
    gravity_x_map -= gravity_up_map * float(gravity_up_map @ gravity_x_map)
    if np.linalg.norm(gravity_x_map) < 1e-6:
        gravity_x_map = np.array((0.0, 1.0, 0.0), dtype=np.float64)
        gravity_x_map -= gravity_up_map * float(gravity_up_map @ gravity_x_map)
    gravity_x_map /= np.linalg.norm(gravity_x_map)
    gravity_y_map = np.cross(gravity_up_map, gravity_x_map)
    gravity_y_map /= np.linalg.norm(gravity_y_map)

    map_T_gravity = np.eye(4, dtype=np.float64)
    map_T_gravity[:3, :3] = np.column_stack(
        (gravity_x_map, gravity_y_map, gravity_up_map)
    )
    if not np.isclose(np.linalg.det(map_T_gravity[:3, :3]), 1.0, atol=1e-8):
        raise ValueError("gravity alignment is not a proper rotation")
    return map_T_gravity


@dataclass(frozen=True)
class HistoricalReplayResult:
    original_estimate_time_ns: int
    consumer_receipt_time_ns: int
    packet: RootStatePacketV2
    payload: bytes


class HistoricalRootStateReplayer:
    """Shift old timestamps into a deterministic valid epoch and run the core."""

    def __init__(
        self,
        *,
        map_T_gravity: np.ndarray,
        imu_T_lidar: np.ndarray,
        calibration_digest: bytes,
        map_frame: str,
        sensor_frame: str,
        target_origin_ns: int = 1_000_000_000_000,
        source_epoch: int = 1,
    ) -> None:
        if target_origin_ns <= 1_000_000:
            raise ValueError("target_origin_ns leaves no calibration-time margin")
        self.target_origin_ns = target_origin_ns
        self._first_original_time_ns: int | None = None
        self._last_original_time_ns: int | None = None
        self._sequence = 0
        calibration = EstimatorCalibration(
            semantics_version="superodom-state-estimation-raw-v1",
            map_frame=map_frame,
            sensor_frame=sensor_frame,
            gravity_frame="gravity_aligned",
            initialization_time_ns=target_origin_ns - 1_000_000,
            map_T_gravity=np.asarray(map_T_gravity, dtype=np.float64),
            imu_T_lidar=np.asarray(imu_T_lidar, dtype=np.float64),
            calibration_digest=calibration_digest,
            valid=True,
        )
        self._core = BridgeCore(
            calibration=calibration,
            kinematics=PelvisKinematics(),
            source_epoch=source_epoch,
            allowed_calibration_digests={calibration_digest},
            max_joint_gap_ns=10_000_000,
            max_correction_age_ns=150_000_000,
            max_health_age_ns=20_000_000,
        )

    def replay(self, observation: EstimatorObservation) -> HistoricalReplayResult:
        original_time_ns = observation.estimate_time_ns
        if self._last_original_time_ns is not None and original_time_ns <= self._last_original_time_ns:
            raise ValueError("historical estimate timestamps must be strictly increasing")
        if self._first_original_time_ns is None:
            self._first_original_time_ns = original_time_ns
        self._last_original_time_ns = original_time_ns
        shifted_time_ns = self.target_origin_ns + (
            original_time_ns - self._first_original_time_ns
        )
        self._sequence += 1

        evidence_receipt_ns = shifted_time_ns + 500_000
        publish_time_ns = shifted_time_ns + 1_000_000
        consumer_receipt_time_ns = shifted_time_ns + 1_500_000
        zeros = (0.0,) * len(CANONICAL_G1_JOINT_NAMES)
        self._core.append_joint(
            TimedJointSample(
                stamp_ns=shifted_time_ns,
                receipt_ns=evidence_receipt_ns,
                names=CANONICAL_G1_JOINT_NAMES,
                position=zeros,
                velocity=zeros,
                sequence=self._sequence,
                source_epoch=1,
            )
        )
        self._core.update_correction(
            CorrectionSample(
                stamp_ns=shifted_time_ns,
                receipt_time_ns=shifted_time_ns + 250_000,
                covariance_diagonal=UNKNOWN_COVARIANCE_DIAGONAL,
            )
        )
        self._core.update_health(
            EstimatorHealthSample(
                receipt_time_ns=evidence_receipt_ns,
                healthy=True,
            )
        )
        shifted_observation = EstimatorObservation(
            estimate_time_ns=shifted_time_ns,
            receipt_time_ns=evidence_receipt_ns,
            frame_id=observation.frame_id,
            child_frame_id=observation.child_frame_id,
            map_T_lidar=observation.map_T_lidar,
            imu_linear_velocity_imu=observation.imu_linear_velocity_imu,
            imu_angular_velocity_imu=observation.imu_angular_velocity_imu,
            estimator_healthy=observation.estimator_healthy,
        )
        packet = self._core.build_packet(
            shifted_observation,
            publish_time_ns=publish_time_ns,
        )
        return HistoricalReplayResult(
            original_estimate_time_ns=original_time_ns,
            consumer_receipt_time_ns=consumer_receipt_time_ns,
            packet=packet,
            payload=serialize_root_state_v2(packet),
        )
