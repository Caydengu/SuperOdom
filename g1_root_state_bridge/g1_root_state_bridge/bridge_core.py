"""Pure, fail-closed SuperOdometry-to-pelvis bridge core.

SuperOdometry's current ``/state_estimation`` contract is intentionally
represented verbosely here:

* pose is ``map_T_lidar`` for the physical Mid-360 LiDAR origin;
* linear velocity is for the IMU origin and expressed in the IMU frame;
* angular velocity is expressed in the IMU frame; and
* the map axes follow the physical sensor at startup and are not necessarily
  z-up.

The normalizer first moves the twist to the LiDAR origin, then applies the
latched startup map-to-gravity alignment.  Dynamic pelvis recovery happens
only after a timestamp-bracketed G1 joint sample is available.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np

from g1_root_state_bridge.joint_buffer import (
    JointBuffer,
    JointSynchronizationError,
)
from g1_root_state_bridge.joint_contract import TimedJointSample
from g1_root_state_bridge.kinematics import (
    KinematicsContractError,
    PelvisState,
    _invert_transform,
    _validated_transform,
    _validated_vector,
)
from g1_root_state_bridge.odometry_health import (
    OdometryHealthConfig,
    OdometryHealthGate,
    OdometryHealthState,
)
from g1_root_state_bridge.orientation_fusion import (
    OrientationFusionConfig,
    OrientationFusionResult,
    PelvisOrientationFusion,
)
from g1_root_state_bridge.protocol import RootStateHealth, RootStatePacketV2
from g1_root_state_bridge.root_imu_buffer import (
    RootImuBuffer,
    RootImuSynchronizationError,
)
from g1_root_state_bridge.root_imu_contract import TimedRootImuSample


ESTIMATOR_SEMANTICS_VERSION = "superodom-state-estimation-raw-v1"
UNKNOWN_COVARIANCE_DIAGONAL = (1_000_000.0,) * 6


class BridgeCoreError(ValueError):
    """The bridge cannot produce a trustworthy state for this observation."""


@dataclass(frozen=True)
class EstimatorCalibration:
    semantics_version: str
    map_frame: str
    sensor_frame: str
    gravity_frame: str
    initialization_time_ns: int
    map_T_gravity: np.ndarray
    imu_T_lidar: np.ndarray
    calibration_digest: bytes
    valid: bool


@dataclass(frozen=True)
class EstimatorObservation:
    estimate_time_ns: int
    receipt_time_ns: int
    frame_id: str
    child_frame_id: str
    map_T_lidar: np.ndarray
    imu_linear_velocity_imu: np.ndarray
    imu_angular_velocity_imu: np.ndarray
    estimator_healthy: bool


@dataclass(frozen=True)
class NormalizedLidarState:
    estimate_time_ns: int
    receipt_time_ns: int
    world_T_lidar: np.ndarray
    lidar_linear_velocity_world: np.ndarray
    lidar_angular_velocity_world: np.ndarray
    estimator_healthy: bool


@dataclass(frozen=True)
class BridgeBuildResult:
    """One root packet plus the exact synchronized joint sample used by FK."""

    packet: RootStatePacketV2
    joint_sample: TimedJointSample


@dataclass(frozen=True)
class CorrectionSample:
    reference_time_ns: int
    evidence_time_ns: int
    application_time_ns: int
    receipt_time_ns: int
    sequence: int
    reset_id: int
    covariance_diagonal: tuple[float, float, float, float, float, float]

    def __post_init__(self) -> None:
        ordered_times = (
            self.reference_time_ns,
            self.evidence_time_ns,
            self.application_time_ns,
            self.receipt_time_ns,
        )
        if not all(isinstance(value, int) and value >= 0 for value in ordered_times):
            raise BridgeCoreError("correction times must be non-negative integers")
        if tuple(sorted(ordered_times)) != ordered_times:
            raise BridgeCoreError(
                "correction times must satisfy reference <= evidence <= application <= receipt"
            )
        for name, value in (("sequence", self.sequence), ("reset_id", self.reset_id)):
            if not isinstance(value, int) or not 0 <= value <= (1 << 64) - 1:
                raise BridgeCoreError(
                    f"correction {name} must be an unsigned 64-bit integer"
                )
        if len(self.covariance_diagonal) != 6:
            raise BridgeCoreError("correction covariance must contain six values")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in self.covariance_diagonal
        ):
            raise BridgeCoreError("correction covariance must be finite and non-negative")


@dataclass(frozen=True)
class EstimatorHealthSample:
    receipt_time_ns: int
    healthy: bool

    def __post_init__(self) -> None:
        if not isinstance(self.receipt_time_ns, int) or self.receipt_time_ns < 0:
            raise BridgeCoreError("health receipt time must be a non-negative integer")
        if not isinstance(self.healthy, bool):
            raise BridgeCoreError("health value must be boolean")


class PelvisKinematicsLike(Protocol):
    def compute(
        self,
        *,
        world_T_observed: np.ndarray,
        observed_linear_velocity_world: np.ndarray,
        observed_angular_velocity_world: np.ndarray,
        joint_sample: TimedJointSample,
        physical_sensor_T_observed: np.ndarray,
    ) -> PelvisState: ...


def _bridge_transform(value: np.ndarray, name: str) -> np.ndarray:
    try:
        return _validated_transform(value, name)
    except KinematicsContractError as error:
        raise BridgeCoreError(f"{name} transform is invalid: {error}") from error


def _bridge_vector(value: np.ndarray, name: str) -> np.ndarray:
    try:
        return _validated_vector(value, name)
    except KinematicsContractError as error:
        raise BridgeCoreError(str(error)) from error


def _quaternion_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert one proper rotation to a normalized, sign-canonical quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal_index = int(np.argmax(np.diag(matrix)))
        if diagonal_index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif diagonal_index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray((w, x, y, z), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return tuple(float(value) for value in quaternion)


class SuperOdometryObservationNormalizer:
    """Normalize the exact current SuperOdometry odometry semantics."""

    def __init__(self, calibration: EstimatorCalibration):
        if calibration.semantics_version != ESTIMATOR_SEMANTICS_VERSION:
            raise BridgeCoreError(
                "unsupported estimator semantics version: "
                f"{calibration.semantics_version!r}"
            )
        if not calibration.map_frame or not calibration.sensor_frame:
            raise BridgeCoreError("calibration frame names must be non-empty")
        if not calibration.gravity_frame:
            raise BridgeCoreError("gravity frame name must be non-empty")
        if (
            not isinstance(calibration.initialization_time_ns, int)
            or calibration.initialization_time_ns < 0
        ):
            raise BridgeCoreError("calibration initialization time is invalid")
        if (
            not isinstance(calibration.calibration_digest, bytes)
            or len(calibration.calibration_digest) != 32
        ):
            raise BridgeCoreError("calibration digest must contain 32 raw bytes")
        self.calibration = calibration
        self._map_T_gravity = _bridge_transform(
            calibration.map_T_gravity, "map_T_gravity"
        )
        if not np.allclose(self._map_T_gravity[:3, 3], 0.0, atol=1e-10):
            raise BridgeCoreError("map_T_gravity transform must have zero translation")
        self._gravity_T_map = _invert_transform(self._map_T_gravity)
        self._imu_T_lidar = _bridge_transform(
            calibration.imu_T_lidar, "imu_T_lidar"
        )

    def normalize(self, observation: EstimatorObservation) -> NormalizedLidarState:
        if observation.frame_id != self.calibration.map_frame:
            raise BridgeCoreError(
                f"estimator frame mismatch: {observation.frame_id!r} != "
                f"{self.calibration.map_frame!r}"
            )
        if observation.child_frame_id != self.calibration.sensor_frame:
            raise BridgeCoreError(
                f"estimator sensor frame mismatch: {observation.child_frame_id!r} != "
                f"{self.calibration.sensor_frame!r}"
            )
        if (
            not isinstance(observation.estimate_time_ns, int)
            or observation.estimate_time_ns < 0
        ):
            raise BridgeCoreError("estimate time must be a non-negative integer")
        if (
            not isinstance(observation.receipt_time_ns, int)
            or observation.receipt_time_ns < 0
        ):
            raise BridgeCoreError("estimate receipt time must be a non-negative integer")

        map_T_lidar = _bridge_transform(observation.map_T_lidar, "map_T_lidar")
        imu_linear_velocity_imu = _bridge_vector(
            observation.imu_linear_velocity_imu, "imu_linear_velocity_imu"
        )
        imu_angular_velocity_imu = _bridge_vector(
            observation.imu_angular_velocity_imu, "imu_angular_velocity_imu"
        )

        # /state_estimation poses the LiDAR but publishes the IMU-origin twist.
        map_T_imu = map_T_lidar @ _invert_transform(self._imu_T_lidar)
        map_R_imu = map_T_imu[:3, :3]
        imu_linear_velocity_map = map_R_imu @ imu_linear_velocity_imu
        angular_velocity_map = map_R_imu @ imu_angular_velocity_imu
        imu_to_lidar_map = map_R_imu @ self._imu_T_lidar[:3, 3]
        lidar_linear_velocity_map = imu_linear_velocity_map + np.cross(
            angular_velocity_map, imu_to_lidar_map
        )

        gravity_R_map = self._gravity_T_map[:3, :3]
        return NormalizedLidarState(
            estimate_time_ns=observation.estimate_time_ns,
            receipt_time_ns=observation.receipt_time_ns,
            world_T_lidar=self._gravity_T_map @ map_T_lidar,
            lidar_linear_velocity_world=gravity_R_map @ lidar_linear_velocity_map,
            lidar_angular_velocity_world=gravity_R_map @ angular_velocity_map,
            estimator_healthy=observation.estimator_healthy,
        )


class BridgeCore:
    """Synchronize estimator evidence and emit one V2 pelvis packet at a time."""

    def __init__(
        self,
        *,
        calibration: EstimatorCalibration,
        kinematics: PelvisKinematicsLike,
        source_epoch: int,
        allowed_calibration_digests: set[bytes],
        max_joint_gap_ns: int,
        max_correction_input_age_ns: int,
        max_correction_age_ns: int,
        max_health_age_ns: int,
        joint_buffer_capacity: int = 4096,
        odometry_health_config: OdometryHealthConfig | None = None,
        orientation_fusion_config: OrientationFusionConfig | None = None,
        max_root_imu_gap_ns: int = 5_000_000,
    ):
        if not isinstance(source_epoch, int) or not 0 <= source_epoch <= (1 << 64) - 1:
            raise BridgeCoreError("source epoch must be an unsigned 64-bit integer")
        if min(
            max_joint_gap_ns,
            max_correction_input_age_ns,
            max_correction_age_ns,
            max_health_age_ns,
        ) < 0:
            raise BridgeCoreError("bridge age and gap thresholds must be non-negative")
        if max_correction_input_age_ns > max_correction_age_ns:
            raise BridgeCoreError(
                "correction input-age threshold cannot exceed hold-age threshold"
            )
        if max_root_imu_gap_ns < 0:
            raise BridgeCoreError("root IMU gap threshold must be non-negative")
        if any(len(digest) != 32 for digest in allowed_calibration_digests):
            raise BridgeCoreError("allowlisted calibration digests must contain 32 bytes")
        self.calibration = calibration
        self.normalizer = SuperOdometryObservationNormalizer(calibration)
        self.kinematics = kinematics
        self.source_epoch = source_epoch
        self.allowed_calibration_digests = frozenset(allowed_calibration_digests)
        self.max_joint_gap_ns = max_joint_gap_ns
        self.max_correction_input_age_ns = max_correction_input_age_ns
        self.max_correction_age_ns = max_correction_age_ns
        self.max_health_age_ns = max_health_age_ns
        self._joint_buffer = JointBuffer(capacity=joint_buffer_capacity)
        self._corrections: deque[CorrectionSample] = deque(maxlen=1024)
        self._health: EstimatorHealthSample | None = None
        self._sequence = 0
        self._physical_sensor_T_observed = np.eye(4)
        self._odometry_health_gate = (
            None
            if odometry_health_config is None
            else OdometryHealthGate(odometry_health_config)
        )
        self.last_odometry_health: OdometryHealthState | None = None
        self.max_root_imu_gap_ns = max_root_imu_gap_ns
        self._root_imu_buffer = RootImuBuffer(capacity=joint_buffer_capacity)
        self._orientation_fusion = (
            None
            if orientation_fusion_config is None
            else PelvisOrientationFusion(orientation_fusion_config)
        )
        self.last_orientation_fusion: OrientationFusionResult | None = None

    def append_joint(self, sample: TimedJointSample) -> None:
        self._joint_buffer.append(sample)

    def append_root_imu(self, sample: TimedRootImuSample) -> None:
        self._root_imu_buffer.append(sample)

    def update_correction(self, sample: CorrectionSample) -> None:
        if self._corrections:
            latest = self._corrections[-1]
            if sample.sequence <= latest.sequence:
                raise BridgeCoreError("correction sequence is not strictly increasing")
            if sample.evidence_time_ns <= latest.evidence_time_ns:
                raise BridgeCoreError("correction evidence time is not strictly increasing")
            if sample.receipt_time_ns < latest.receipt_time_ns:
                raise BridgeCoreError("correction receipt time is not monotonic")
        self._corrections.append(sample)

    def update_health(self, sample: EstimatorHealthSample) -> None:
        if self._health is not None and sample.receipt_time_ns <= self._health.receipt_time_ns:
            raise BridgeCoreError("health receipt time is not strictly increasing")
        self._health = sample

    def reset(self, *, source_epoch: int) -> None:
        if not isinstance(source_epoch, int) or not self.source_epoch < source_epoch <= (1 << 64) - 1:
            raise BridgeCoreError("reset source epoch must strictly increase")
        self.source_epoch = source_epoch
        self._sequence = 0
        self._joint_buffer.clear()
        self._root_imu_buffer.clear()
        self._corrections.clear()
        self._health = None
        if self._odometry_health_gate is not None:
            self._odometry_health_gate.reset()
        self.last_odometry_health = None
        if self._orientation_fusion is not None:
            self._orientation_fusion.reset()
        self.last_orientation_fusion = None

    def _correction_at(
        self,
        *,
        estimate_time_ns: int,
        observation_receipt_time_ns: int,
    ) -> CorrectionSample | None:
        for sample in reversed(self._corrections):
            if (
                sample.evidence_time_ns <= estimate_time_ns
                and sample.application_time_ns <= observation_receipt_time_ns
            ):
                return sample
        return None

    def build_packet(
        self,
        observation: EstimatorObservation,
        *,
        publish_time_ns: int,
    ) -> RootStatePacketV2:
        return self.build_packet_with_evidence(
            observation,
            publish_time_ns=publish_time_ns,
        ).packet

    def build_packet_with_evidence(
        self,
        observation: EstimatorObservation,
        *,
        publish_time_ns: int,
    ) -> BridgeBuildResult:
        if not isinstance(publish_time_ns, int) or publish_time_ns < 0:
            raise BridgeCoreError("publish time must be a non-negative integer")
        normalized = self.normalizer.normalize(observation)
        try:
            joint_match = self._joint_buffer.synchronize(
                normalized.estimate_time_ns,
                max_gap_ns=self.max_joint_gap_ns,
            )
        except JointSynchronizationError as error:
            raise BridgeCoreError(f"joint synchronization failed: {error}") from error

        try:
            pelvis = self.kinematics.compute(
                world_T_observed=normalized.world_T_lidar,
                observed_linear_velocity_world=normalized.lidar_linear_velocity_world,
                observed_angular_velocity_world=normalized.lidar_angular_velocity_world,
                joint_sample=joint_match.sample,
                physical_sensor_T_observed=self._physical_sensor_T_observed,
            )
        except (KinematicsContractError, ValueError) as error:
            raise BridgeCoreError(f"pelvis kinematics failed: {error}") from error

        world_T_pelvis = _bridge_transform(pelvis.world_T_pelvis, "world_T_pelvis")
        pelvis_linear = _bridge_vector(
            pelvis.linear_velocity_world, "pelvis_linear_velocity_world"
        )
        pelvis_angular = _bridge_vector(
            pelvis.angular_velocity_world, "pelvis_angular_velocity_world"
        )
        root_imu_sync_valid = False
        root_orientation_fused = False
        if self._orientation_fusion is not None:
            try:
                root_imu_match = self._root_imu_buffer.synchronize(
                    normalized.estimate_time_ns,
                    max_gap_ns=self.max_root_imu_gap_ns,
                )
            except RootImuSynchronizationError as error:
                raise BridgeCoreError(
                    f"root IMU synchronization failed: {error}"
                ) from error
            root_imu_sync_valid = (
                root_imu_match.maximum_bracket_gap_ns <= self.max_root_imu_gap_ns
            )
            self.last_orientation_fusion = self._orientation_fusion.update(
                source_time_ns=normalized.estimate_time_ns,
                local_world_R_pelvis=world_T_pelvis[:3, :3],
                root_imu=root_imu_match.sample,
            )
            if self.last_orientation_fusion.healthy:
                world_T_pelvis[:3, :3] = (
                    self.last_orientation_fusion.world_R_pelvis
                )
                pelvis_angular = self.last_orientation_fusion.angular_velocity_world
                root_orientation_fused = True
        pelvis_quaternion = _quaternion_wxyz(world_T_pelvis[:3, :3])
        if self._odometry_health_gate is not None:
            try:
                self.last_odometry_health = self._odometry_health_gate.update(
                    source_time_ns=normalized.estimate_time_ns,
                    position_xyz_m=world_T_pelvis[:3, 3],
                    quaternion_wxyz=pelvis_quaternion,
                )
            except ValueError as error:
                raise BridgeCoreError(
                    f"odometry plausibility check failed: {error}"
                ) from error
        correction = self._correction_at(
            estimate_time_ns=normalized.estimate_time_ns,
            observation_receipt_time_ns=normalized.receipt_time_ns,
        )

        health_flags = RootStateHealth(0)
        health_sample = self._health
        health_fresh = (
            health_sample is not None
            and 0 <= publish_time_ns - health_sample.receipt_time_ns <= self.max_health_age_ns
        )
        plausibility_healthy = (
            self.last_odometry_health is None
            or self.last_odometry_health.healthy
        )
        if (
            normalized.estimator_healthy
            and health_fresh
            and health_sample.healthy
            and plausibility_healthy
            and (
                self._orientation_fusion is None
                or root_orientation_fused
            )
        ):
            health_flags |= RootStateHealth.ESTIMATOR_HEALTHY
        health_flags |= RootStateHealth.FINITE_POSE
        if joint_match.maximum_bracket_gap_ns <= self.max_joint_gap_ns:
            health_flags |= RootStateHealth.JOINT_SYNC_VALID
        correction_fresh = (
            correction is not None
            and 0
            <= correction.receipt_time_ns - correction.evidence_time_ns
            <= self.max_correction_input_age_ns
            and 0
            <= publish_time_ns - correction.evidence_time_ns
            <= self.max_correction_age_ns
        )
        if correction_fresh:
            health_flags |= RootStateHealth.CORRECTION_FRESH
        if (
            self.calibration.valid
            and self.calibration.calibration_digest in self.allowed_calibration_digests
        ):
            health_flags |= RootStateHealth.CALIBRATION_VALID

        clock_valid = (
            self.calibration.initialization_time_ns <= normalized.estimate_time_ns
            and normalized.estimate_time_ns <= normalized.receipt_time_ns <= publish_time_ns
            and joint_match.sample.receipt_ns <= publish_time_ns
            and (correction is None or correction.receipt_time_ns <= publish_time_ns)
            and (health_sample is None or health_sample.receipt_time_ns <= publish_time_ns)
        )
        if clock_valid:
            health_flags |= RootStateHealth.CLOCK_VALID
        if root_imu_sync_valid:
            health_flags |= RootStateHealth.ROOT_IMU_SYNC_VALID
        if root_orientation_fused:
            health_flags |= RootStateHealth.ROOT_ORIENTATION_FUSED

        self._sequence += 1
        correction_time_ns = 0 if correction is None else correction.evidence_time_ns
        covariance = (
            UNKNOWN_COVARIANCE_DIAGONAL
            if correction is None
            else correction.covariance_diagonal
        )
        return BridgeBuildResult(
            packet=RootStatePacketV2(
                sequence=self._sequence,
                source_epoch=self.source_epoch,
                estimate_time_ns=normalized.estimate_time_ns,
                publish_time_ns=publish_time_ns,
                correction_time_ns=correction_time_ns,
                joint_time_ns=joint_match.representative_time_ns,
                joint_sync_gap_ns=joint_match.signed_gap_ns,
                health_flags=health_flags,
                position=tuple(
                    float(value) for value in world_T_pelvis[:3, 3]
                ),
                quaternion_wxyz=pelvis_quaternion,
                linear_velocity=tuple(
                    float(value) for value in pelvis_linear
                ),
                angular_velocity=tuple(
                    float(value) for value in pelvis_angular
                ),
                covariance_diagonal=covariance,
                calibration_digest=self.calibration.calibration_digest,
            ),
            joint_sample=joint_match.sample,
        )
