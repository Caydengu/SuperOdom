"""Generate matched sensor, dynamic-FK, and pelvis-IMU AMO pose treatments."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from g1_root_state_bridge.amo_dataset import LowStateCapture
from g1_root_state_bridge.joint_contract import (
    CANONICAL_G1_JOINT_NAMES,
    TimedJointSample,
)
from g1_root_state_bridge.kinematics import PelvisKinematics
from g1_root_state_bridge.orientation_fusion import (
    OrientationFusionConfig,
    PelvisOrientationFusion,
    gravity_aligned_heading,
)
from g1_root_state_bridge.root_imu_contract import TimedRootImuSample


class AmoTreatmentError(ValueError):
    """A source track cannot be transformed under the matched-input contract."""


@dataclass(frozen=True)
class OdometryTrackLoadResult:
    """Causally admitted odometry plus explicit stale-message diagnostics."""

    records: list[dict[str, object]]
    total_records: int
    rejected_stale_records: tuple[dict[str, int], ...]

    @property
    def health(self) -> dict[str, object]:
        return {
            "ordering_contract": "bag_order_with_nonincreasing_source_timestamps_rejected",
            "total_records": self.total_records,
            "admitted_records": len(self.records),
            "rejected_stale_count": len(self.rejected_stale_records),
            "rejected_stale_fraction": (
                len(self.rejected_stale_records) / self.total_records
                if self.total_records
                else 0.0
            ),
            "rejected_stale_records": list(self.rejected_stale_records),
        }


def rotation_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm((w, x, y, z)))
    if not np.isfinite(norm) or norm <= 0.0:
        raise AmoTreatmentError("odometry quaternion is invalid")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def xyzw_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray((x, y, z, w), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def load_odometry_track(
    path: str | Path,
    *,
    maximum_stale_fraction: float = 0.001,
) -> OdometryTrackLoadResult:
    """Load in recorded arrival order and reject stale source timestamps.

    A live causal consumer cannot sort a late estimator output into the past.
    Isolated late messages are therefore rejected and surfaced as health
    events.  Persistent clock resets fail closed via ``maximum_stale_fraction``.
    """

    records: list[dict[str, object]] = []
    rejected: list[dict[str, int]] = []
    total_records = 0
    last_source_time_ns: int | None = None
    last_receipt_time_ns: int | None = None
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            if payload.get("kind") == "metadata":
                continue
            if payload.get("kind") != "odometry":
                raise AmoTreatmentError("track contains an unsupported record")
            if payload.get("child_frame_id") != "sensor":
                raise AmoTreatmentError(
                    "source odometry must describe the sensor frame"
                )
            total_records += 1
            source_time_ns = int(payload["source_time_ns"])
            receipt_time_ns = int(payload["receipt_time_ns"])
            if (
                last_receipt_time_ns is not None
                and receipt_time_ns < last_receipt_time_ns
            ):
                raise AmoTreatmentError("odometry receipt timestamps moved backwards")
            last_receipt_time_ns = receipt_time_ns
            if (
                last_source_time_ns is not None
                and source_time_ns <= last_source_time_ns
            ):
                rejected.append(
                    {
                        "record_index": total_records - 1,
                        "source_time_ns": source_time_ns,
                        "receipt_time_ns": receipt_time_ns,
                        "previous_admitted_source_time_ns": last_source_time_ns,
                        "rollback_ns": last_source_time_ns - source_time_ns,
                    }
                )
                continue
            records.append(payload)
            last_source_time_ns = source_time_ns
    if not records:
        raise AmoTreatmentError("source odometry track is empty")
    stale_fraction = len(rejected) / total_records
    if stale_fraction > maximum_stale_fraction:
        raise AmoTreatmentError(
            f"stale odometry fraction {stale_fraction:.6f} exceeds "
            f"{maximum_stale_fraction:.6f}; possible persistent estimator reset"
        )
    return OdometryTrackLoadResult(
        records=records,
        total_records=total_records,
        rejected_stale_records=tuple(rejected),
    )


def _nearest_indices(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.searchsorted(reference, query, side="left")
    right = np.clip(right, 0, reference.size - 1)
    left = np.clip(right - 1, 0, reference.size - 1)
    choose_left = np.abs(query - reference[left]) <= np.abs(reference[right] - query)
    return np.where(choose_left, left, right)


def generate_treatments(
    source_records: list[dict[str, object]],
    lowstate: LowStateCapture,
    *,
    maximum_joint_age_ms: float = 10.0,
    minimum_joint_coverage: float = 0.99,
) -> Iterator[dict[str, object]]:
    source_times = np.asarray(
        [record["source_time_ns"] for record in source_records], dtype=np.int64
    )
    admissible = (source_times >= lowstate.robot_stamp_ns[0]) & (
        source_times <= lowstate.robot_stamp_ns[-1]
    )
    source_records = [
        record for record, keep in zip(source_records, admissible) if keep
    ]
    source_times = source_times[admissible]
    indices = _nearest_indices(lowstate.robot_stamp_ns, source_times)
    ages_ns = np.abs(source_times - lowstate.robot_stamp_ns[indices])
    joint_admissible = ages_ns <= maximum_joint_age_ms * 1e6
    coverage = float(np.mean(joint_admissible)) if joint_admissible.size else 0.0
    if coverage < minimum_joint_coverage:
        raise AmoTreatmentError(
            f"lowstate match coverage {coverage:.6f} is below {minimum_joint_coverage:.6f}"
        )
    source_records = [
        record for record, keep in zip(source_records, joint_admissible) if keep
    ]
    source_times = source_times[joint_admissible]
    indices = indices[joint_admissible]
    ages_ns = ages_ns[joint_admissible]

    kinematics = PelvisKinematics()
    identity = np.eye(4, dtype=np.float64)
    fusions = {
        "superodom_dynamic_fk_pelvis_imu_sign_pos": PelvisOrientationFusion(
            OrientationFusionConfig(gyro_z_sign=1.0)
        ),
        "superodom_dynamic_fk_pelvis_imu_sign_neg": PelvisOrientationFusion(
            OrientationFusionConfig(gyro_z_sign=-1.0)
        ),
    }
    for record, source_time_ns, low_index, lowstate_age_ns in zip(
        source_records, source_times, indices, ages_ns
    ):
        source_time = int(source_time_ns)
        position = np.asarray(record["position_xyz_m"], dtype=np.float64)
        sensor_rotation = rotation_from_xyzw(np.asarray(record["quaternion_xyzw"]))
        world_T_sensor = np.eye(4, dtype=np.float64)
        world_T_sensor[:3, :3] = sensor_rotation
        world_T_sensor[:3, 3] = position
        joint = TimedJointSample(
            stamp_ns=source_time,
            receipt_ns=source_time,
            names=CANONICAL_G1_JOINT_NAMES,
            position=tuple(
                float(value) for value in lowstate.joint_position[low_index]
            ),
            velocity=tuple(
                float(value) for value in lowstate.joint_velocity[low_index]
            ),
            sequence=int(lowstate.sequence[low_index]),
            source_epoch=int(lowstate.source_epoch[low_index]),
        )
        world_T_pelvis = kinematics.recover_pelvis_pose(
            world_T_observed=world_T_sensor,
            joint_sample=joint,
            physical_sensor_T_observed=identity,
        )
        event_realtime_ns = round(
            float(lowstate.clock.map_ns(np.asarray([source_time]))[0])
        )
        common = {
            "schema": "g1_amo_localization_treatment_track_v1",
            "kind": "pose",
            "source_time_ns": source_time,
            "event_realtime_ns": event_realtime_ns,
            "source_lowstate_sequence": int(lowstate.sequence[low_index]),
            "source_lowstate_age_ns": int(lowstate_age_ns),
            "frame_id": str(record["frame_id"]),
        }
        yield {
            **common,
            "treatment": "superodom_sensor",
            "child_frame_id": "sensor",
            "position_xyz_m": [float(value) for value in position],
            "quaternion_xyzw": [float(value) for value in record["quaternion_xyzw"]],
        }
        yield {
            **common,
            "treatment": "superodom_dynamic_fk_pelvis",
            "child_frame_id": "pelvis",
            "position_xyz_m": [float(value) for value in world_T_pelvis[:3, 3]],
            "quaternion_xyzw": list(xyzw_from_rotation(world_T_pelvis[:3, :3])),
        }
        yield {
            **common,
            "treatment": "superodom_dynamic_fk_position_lio_heading_root_gravity",
            "child_frame_id": "pelvis",
            "position_xyz_m": [float(value) for value in world_T_pelvis[:3, 3]],
            "quaternion_xyzw": list(
                xyzw_from_rotation(
                    gravity_aligned_heading(
                        heading_rotation=sensor_rotation,
                        gravity_quaternion_wxyz=tuple(
                            float(value)
                            for value in lowstate.imu_quaternion_wxyz[low_index]
                        ),
                    )
                )
            ),
            "orientation_fusion_healthy": True,
            "orientation_fusion_reason": "stateless_lio_heading_root_gravity",
            "yaw_innovation_rad": 0.0,
        }
        root_imu = TimedRootImuSample(
            stamp_ns=source_time,
            receipt_ns=source_time,
            quaternion_wxyz=tuple(
                float(value) for value in lowstate.imu_quaternion_wxyz[low_index]
            ),
            angular_velocity=tuple(
                float(value) for value in lowstate.imu_gyroscope[low_index]
            ),
            linear_acceleration=tuple(
                float(value) for value in lowstate.imu_accelerometer[low_index]
            ),
            sequence=int(lowstate.sequence[low_index]),
            source_epoch=int(lowstate.source_epoch[low_index]),
        )
        for treatment, fusion in fusions.items():
            fused = fusion.update(
                source_time_ns=source_time,
                local_world_R_pelvis=world_T_pelvis[:3, :3],
                root_imu=root_imu,
            )
            yield {
                **common,
                "treatment": treatment,
                "child_frame_id": "pelvis",
                "position_xyz_m": [float(value) for value in world_T_pelvis[:3, 3]],
                "quaternion_xyzw": list(xyzw_from_rotation(fused.world_R_pelvis)),
                "orientation_fusion_healthy": bool(fused.healthy),
                "orientation_fusion_reason": fused.reason,
                "yaw_innovation_rad": float(fused.yaw_innovation_rad),
            }
