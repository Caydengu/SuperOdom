"""Replay/live-shared selected localization pipeline for G1-4123."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import time
from typing import Protocol

import numpy as np

from g1_root_state_bridge.joint_buffer import JointBuffer, JointSynchronizationError
from g1_root_state_bridge.joint_contract import CANONICAL_G1_JOINT_NAMES, TimedJointSample
from g1_root_state_bridge.kiss_pelvis_local_odometry import (
    KissPelvisHealthConfig,
    KissPelvisObservation,
    KissPelvisPacketBuilder,
)
from g1_root_state_bridge.kiss_registration import RegistrationResult
from g1_root_state_bridge.livox_deskew import LivoxDeskewError, gyro_deskew_scan
from g1_root_state_bridge.protocol import RootStatePacketV2, serialize_root_state_v2
from g1_root_state_bridge.waist_kinematics import (
    pelvis_T_mid360,
    sensor_local_pose_to_pelvis_local_pose,
)


class LiveLocalizationError(RuntimeError):
    """The live-shaped pipeline cannot produce a trustworthy state."""


class RegistrationBackend(Protocol):
    def register(self, points_xyz_m: np.ndarray) -> RegistrationResult: ...


@dataclass(frozen=True)
class LiveLocalizationConfig:
    gyro_bias_radps: tuple[float, float, float]
    sensor_z_to_torso_yaw_sign: float = -1.0
    maximum_imu_gap_ns: int = 25_000_000
    maximum_joint_gap_ns: int = 10_000_000
    maximum_correction_age_ns: int = 150_000_000
    imu_buffer_capacity: int = 4_096
    covariance_diagonal: tuple[float, float, float, float, float, float] = (
        0.0025,
        0.0025,
        0.01,
        0.001,
        0.001,
        0.003,
    )

    def __post_init__(self) -> None:
        bias = np.asarray(self.gyro_bias_radps, dtype=np.float64)
        if bias.shape != (3,) or not np.all(np.isfinite(bias)):
            raise LiveLocalizationError("gyro bias must contain three finite values")
        if self.sensor_z_to_torso_yaw_sign not in (-1.0, 1.0):
            raise LiveLocalizationError("sensor yaw sign must be exactly -1 or +1")
        if min(
            self.maximum_imu_gap_ns,
            self.maximum_joint_gap_ns,
            self.maximum_correction_age_ns,
            self.imu_buffer_capacity,
        ) <= 0:
            raise LiveLocalizationError("buffer and timing limits must be positive")

    @property
    def calibration_digest(self) -> bytes:
        payload = json.dumps(
            {
                "schema": "g1_4123_kiss_live_localization_calibration_v1",
                "gyro_bias_radps": self.gyro_bias_radps,
                "sensor_z_to_torso_yaw_sign": self.sensor_z_to_torso_yaw_sign,
                "waist_chain": "g1_pelvis_mid360_kinematics_v1",
                "heading": "livox_torso_navigation_yaw_without_waist_subtraction",
                "point_time": "float32_seconds_relative_to_scan_header",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).digest()


@dataclass(frozen=True)
class PipelineOutput:
    packet: RootStatePacketV2
    payload: bytes
    local_T_pelvis: np.ndarray
    registered_points_local_xyz_m: np.ndarray
    registered_cloud_xyz32: bytes
    adaptive_threshold: float
    deskew_angular_excursion_deg: float
    source_joint_sequence: int
    source_joint_time_ns: int
    stage_runtime_ms: dict[str, float]


class _ImuYawBuffer:
    def __init__(self, *, capacity: int, bias_z: float, sign: float, maximum_gap_ns: int) -> None:
        self._samples: deque[tuple[int, np.ndarray, float]] = deque(maxlen=capacity)
        self._bias_z = float(bias_z)
        self._sign = float(sign)
        self._maximum_gap_ns = maximum_gap_ns
        self._torso_yaw = 0.0

    def clear(self) -> None:
        self._samples.clear()
        self._torso_yaw = 0.0

    def append(self, time_ns: int, angular_velocity_radps: np.ndarray) -> None:
        angular = np.asarray(angular_velocity_radps, dtype=np.float64)
        if not isinstance(time_ns, int) or time_ns <= 0 or angular.shape != (3,) or not np.all(np.isfinite(angular)):
            raise LiveLocalizationError("IMU sample is invalid")
        if self._samples:
            previous_time, previous_angular, _ = self._samples[-1]
            if time_ns <= previous_time:
                raise LiveLocalizationError("IMU timestamp did not increase")
            gap_ns = time_ns - previous_time
            if gap_ns > self._maximum_gap_ns:
                raise LiveLocalizationError("IMU gap exceeds the fail-closed limit")
            previous_rate = self._sign * (previous_angular[2] - self._bias_z)
            current_rate = self._sign * (angular[2] - self._bias_z)
            self._torso_yaw += 0.5 * (previous_rate + current_rate) * gap_ns * 1e-9
        self._samples.append((time_ns, angular.copy(), self._torso_yaw))

    def arrays_for_scan(self, start_time_ns: int, end_time_ns: int) -> tuple[np.ndarray, np.ndarray]:
        if len(self._samples) < 2:
            raise LiveLocalizationError("IMU buffer has fewer than two samples")
        values = list(self._samples)
        time_values = np.asarray([sample[0] for sample in values], dtype=np.int64)
        before = np.searchsorted(time_values, start_time_ns, side="right") - 1
        after = np.searchsorted(time_values, end_time_ns, side="left")
        if before < 0 or after >= len(values):
            raise LiveLocalizationError("IMU buffer does not bracket the scan")
        selected = values[before : after + 1]
        return (
            np.asarray([sample[0] for sample in selected], dtype=np.int64),
            np.asarray([sample[1] for sample in selected], dtype=np.float64),
        )

    def torso_yaw_at(self, query_time_ns: int) -> float:
        values = list(self._samples)
        if len(values) < 2 or not values[0][0] <= query_time_ns <= values[-1][0]:
            raise LiveLocalizationError("IMU yaw buffer does not bracket the query")
        time_ns = np.asarray([sample[0] for sample in values], dtype=np.int64)
        yaw = np.asarray([sample[2] for sample in values], dtype=np.float64)
        origin = int(time_ns[0])
        return float(np.interp((query_time_ns - origin) * 1e-9, (time_ns - origin) * 1e-9, yaw))


def _compose_navigation_pose(native_pelvis_pose: np.ndarray, navigation_yaw: float) -> np.ndarray:
    """Keep KISS/FK position and expose the frozen torso navigation heading.

    The G1-4123 separator showed that rotating each KISS translation increment
    under the replacement yaw made both stress runs worse.  The selected 2D
    robot-vlm contract is deliberately mixed: dynamic-FK pelvis origin x/y/z,
    plus torso-aligned Livox navigation yaw.  Waist yaw is already needed for
    the articulated sensor lever arm; it must not be subtracted from heading.
    """

    pose = np.asarray(native_pelvis_pose, dtype=np.float64)
    output = np.eye(4, dtype=np.float64)
    c_yaw, s_yaw = math.cos(navigation_yaw), math.sin(navigation_yaw)
    output[:3, :3] = np.asarray(
        ((c_yaw, -s_yaw, 0.0), (s_yaw, c_yaw, 0.0), (0.0, 0.0, 1.0))
    )
    output[:3, 3] = pose[:3, 3]
    return output


class SelectedLocalizationPipeline:
    """Stateful fast local lane shared by archive replay and the ROS node."""

    def __init__(self, registration: RegistrationBackend, config: LiveLocalizationConfig) -> None:
        self.registration = registration
        self.config = config
        self.joints = JointBuffer(capacity=4_096)
        self.imu = _ImuYawBuffer(
            capacity=config.imu_buffer_capacity,
            bias_z=config.gyro_bias_radps[2],
            sign=config.sensor_z_to_torso_yaw_sign,
            maximum_gap_ns=config.maximum_imu_gap_ns,
        )
        self.packet_builder = KissPelvisPacketBuilder(
            calibration_digest=config.calibration_digest,
            config=KissPelvisHealthConfig(
                maximum_joint_sync_gap_ns=config.maximum_joint_gap_ns,
                maximum_correction_age_ns=config.maximum_correction_age_ns,
            ),
        )
        self._pelvis0_T_sensor0: np.ndarray | None = None
        self._reference_torso_yaw: float | None = None
        self._source_epoch = 0
        self._registration_initialized = False

    def reset(self, source_epoch: int) -> None:
        if source_epoch < self._source_epoch:
            raise LiveLocalizationError("pipeline source epoch regressed")
        self._source_epoch = source_epoch
        self.joints.clear()
        self.imu.clear()
        self.packet_builder = KissPelvisPacketBuilder(
            calibration_digest=self.config.calibration_digest,
            config=KissPelvisHealthConfig(
                maximum_joint_sync_gap_ns=self.config.maximum_joint_gap_ns,
                maximum_correction_age_ns=self.config.maximum_correction_age_ns,
            ),
        )
        self._pelvis0_T_sensor0 = None
        self._reference_torso_yaw = None
        reset_registration = getattr(self.registration, "reset", None)
        if callable(reset_registration):
            reset_registration()
        self._registration_initialized = False

    def append_imu(self, time_ns: int, angular_velocity_radps: np.ndarray) -> None:
        self.imu.append(time_ns, angular_velocity_radps)

    def append_joint(self, sample: TimedJointSample) -> None:
        self.joints.append(sample)

    def bootstrap_unpublished_scan(self, points_xyz_m: np.ndarray) -> RegistrationResult:
        """Seed KISS once from a partial startup scan without emitting state.

        The frozen research treatment admitted the first raw capture-boundary
        scan into KISS but marked it deskew-invalid.  Making that one-time state
        transition explicit preserves the local-map identity while ensuring no
        packet can claim the scan was healthy.
        """

        if self._registration_initialized:
            raise LiveLocalizationError("raw bootstrap is allowed only before registration")
        result = self.registration.register(points_xyz_m)
        self._registration_initialized = True
        return result

    def process_scan(
        self,
        points_xyz_m: np.ndarray,
        point_relative_time_s: np.ndarray,
        *,
        scan_start_time_ns: int,
        publish_time_ns: int | None = None,
        publish_time_offset_ns: int | None = None,
        clock_valid: bool = True,
        calibration_valid: bool = True,
    ) -> PipelineOutput:
        if publish_time_ns is not None and publish_time_offset_ns is not None:
            raise LiveLocalizationError(
                "publish_time_ns and publish_time_offset_ns are mutually exclusive"
            )
        if publish_time_offset_ns is not None and publish_time_offset_ns < 0:
            raise LiveLocalizationError("publish time offset must be non-negative")
        total_start = time.perf_counter_ns()
        scan_end_time_ns = int(scan_start_time_ns + round(float(np.max(point_relative_time_s)) * 1e9))
        imu_time_ns, imu_angular = self.imu.arrays_for_scan(scan_start_time_ns, scan_end_time_ns)
        stage_start = time.perf_counter_ns()
        try:
            deskewed, estimate_time_ns, excursion_deg = gyro_deskew_scan(
                points_xyz_m,
                point_relative_time_s,
                scan_start_time_ns=scan_start_time_ns,
                imu_time_ns=imu_time_ns,
                imu_angular_velocity_radps=imu_angular,
                gyro_bias_radps=np.asarray(self.config.gyro_bias_radps),
                reference="end",
            )
        except LivoxDeskewError as error:
            raise LiveLocalizationError(str(error)) from error
        deskew_ms = (time.perf_counter_ns() - stage_start) * 1e-6

        stage_start = time.perf_counter_ns()
        registration = self.registration.register(deskewed)
        self._registration_initialized = True
        registration_wall_ms = (time.perf_counter_ns() - stage_start) * 1e-6

        stage_start = time.perf_counter_ns()
        try:
            synchronized = self.joints.synchronize(estimate_time_ns, self.config.maximum_joint_gap_ns)
        except JointSynchronizationError as error:
            raise LiveLocalizationError(str(error)) from error
        waist_yaw, waist_roll, waist_pitch = synchronized.sample.position[12:15]
        pelvis_t_T_sensor_t = pelvis_T_mid360(waist_yaw, waist_roll, waist_pitch)
        if self._pelvis0_T_sensor0 is None:
            self._pelvis0_T_sensor0 = pelvis_t_T_sensor_t.copy()
        native_pelvis_pose = sensor_local_pose_to_pelvis_local_pose(
            registration.sensor0_T_sensor,
            pelvis0_T_sensor0=self._pelvis0_T_sensor0,
            pelvis_t_T_sensor_t=pelvis_t_T_sensor_t,
        )
        torso_yaw = self.imu.torso_yaw_at(estimate_time_ns)
        if self._reference_torso_yaw is None:
            self._reference_torso_yaw = torso_yaw
        navigation_yaw = torso_yaw - self._reference_torso_yaw
        local_T_pelvis = _compose_navigation_pose(native_pelvis_pose, navigation_yaw)
        fusion_ms = (time.perf_counter_ns() - stage_start) * 1e-6

        stage_start = time.perf_counter_ns()
        local_T_sensor = self._pelvis0_T_sensor0 @ registration.sensor0_T_sensor
        registered_points_local = (
            deskewed @ local_T_sensor[:3, :3].T + local_T_sensor[:3, 3]
        )
        cloud_transform_ms = (time.perf_counter_ns() - stage_start) * 1e-6

        stage_start = time.perf_counter_ns()
        registered_cloud_xyz32 = np.ascontiguousarray(
            registered_points_local, dtype="<f4"
        ).tobytes(order="C")
        cloud_pack_ms = (time.perf_counter_ns() - stage_start) * 1e-6

        if publish_time_offset_ns is not None:
            processing_so_far_ns = time.perf_counter_ns() - total_start
            publish_ns = estimate_time_ns + publish_time_offset_ns + processing_so_far_ns
        else:
            publish_ns = time.time_ns() if publish_time_ns is None else int(publish_time_ns)
        if publish_ns < estimate_time_ns:
            raise LiveLocalizationError("publish time predates the source-time estimate")
        stage_start = time.perf_counter_ns()
        observation = KissPelvisObservation(
            source_epoch=self._source_epoch,
            estimate_time_ns=estimate_time_ns,
            correction_time_ns=estimate_time_ns,
            joint_time_ns=synchronized.representative_time_ns,
            publish_time_ns=publish_ns,
            receipt_time_ns=publish_ns,
            local_T_pelvis=local_T_pelvis,
            covariance_diagonal=self.config.covariance_diagonal,
            registration_valid=True,
            deskew_valid=True,
            heading_valid=True,
            calibration_valid=calibration_valid,
            clock_valid=clock_valid,
        )
        packet = self.packet_builder.build(observation)
        payload = serialize_root_state_v2(packet)
        pack_ms = (time.perf_counter_ns() - stage_start) * 1e-6
        total_ms = (time.perf_counter_ns() - total_start) * 1e-6
        return PipelineOutput(
            packet=packet,
            payload=payload,
            local_T_pelvis=local_T_pelvis,
            registered_points_local_xyz_m=registered_points_local,
            registered_cloud_xyz32=registered_cloud_xyz32,
            adaptive_threshold=registration.adaptive_threshold,
            deskew_angular_excursion_deg=excursion_deg,
            source_joint_sequence=synchronized.representative_sequence,
            source_joint_time_ns=synchronized.representative_time_ns,
            stage_runtime_ms={
                "deskew": deskew_ms,
                "registration": registration.runtime_ms,
                "registration_wall": registration_wall_ms,
                "fk_heading": fusion_ms,
                "cloud_transform": cloud_transform_ms,
                "cloud_pack": cloud_pack_ms,
                "pack": pack_ms,
                "total": total_ms,
            },
        )


def joint_sample_from_dynamic_packet(
    packet: object,
    *,
    receipt_time_ns: int,
    mapped_stamp_ns: int | None = None,
) -> TimedJointSample:
    """Adapt one validated DynamicCapturePacketV1 without importing Unitree."""

    return TimedJointSample(
        stamp_ns=(int(packet.robot_stamp_ns) if mapped_stamp_ns is None else mapped_stamp_ns),
        receipt_ns=max(
            int(receipt_time_ns),
            int(packet.robot_stamp_ns) if mapped_stamp_ns is None else mapped_stamp_ns,
        ),
        names=CANONICAL_G1_JOINT_NAMES,
        position=tuple(float(value) for value in packet.joint_position),
        velocity=tuple(float(value) for value in packet.joint_velocity),
        sequence=int(packet.sequence),
        source_epoch=int(packet.source_epoch),
    )
