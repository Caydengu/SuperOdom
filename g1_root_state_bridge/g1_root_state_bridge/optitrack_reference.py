"""Convert raw Motive rigid-body records into calibrated pelvis reference poses."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


RAW_SCHEMA = "g1_optitrack_raw_v1"
CALIBRATION_SCHEMA = "g1_optitrack_pelvis_calibration_v1"
REFERENCE_SCHEMA = "g1_dynamic_reference_v1"


class OptiTrackReferenceError(ValueError):
    """Raised when raw OptiTrack evidence is not safe to call pelvis truth."""


def _tuple_of_floats(
    value: object,
    *,
    expected_length: int,
    field_name: str,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != expected_length:
        raise OptiTrackReferenceError(
            f"{field_name} must contain {expected_length} values"
        )
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise OptiTrackReferenceError(f"{field_name} must be finite")
    return result


def _require_transform(
    payload: Mapping[str, object],
    field_name: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    transform = payload.get(field_name)
    if not isinstance(transform, Mapping):
        raise OptiTrackReferenceError(f"{field_name} must be an object")
    translation = _tuple_of_floats(
        transform.get("translation_m"),
        expected_length=3,
        field_name=f"{field_name}.translation_m",
    )
    quaternion = _tuple_of_floats(
        transform.get("quaternion_wxyz"),
        expected_length=4,
        field_name=f"{field_name}.quaternion_wxyz",
    )
    norm = math.sqrt(sum(value * value for value in quaternion))
    if abs(norm - 1.0) > 1e-3:
        raise OptiTrackReferenceError(
            f"{field_name}.quaternion_wxyz must be unit length"
        )
    return translation, quaternion


@dataclass(frozen=True)
class PelvisReferenceCalibration:
    calibration_id: str
    raw_capture_calibration_id: str
    rigid_body_id: int
    rigid_body_name: str
    target_frame_id: str
    clock_id: str
    clock_mapping_method: str
    clock_mapping_calibration_id: str
    clock_mapping_residual_p95_ns: int
    max_clock_mapping_residual_ns: int
    maximum_marker_error_m: float
    maximum_velocity_interval_ns: int
    motive_world_to_target_translation_m: tuple[float, float, float]
    motive_world_to_target_quaternion_wxyz: tuple[float, float, float, float]
    rigid_body_to_pelvis_translation_m: tuple[float, float, float]
    rigid_body_to_pelvis_quaternion_wxyz: tuple[float, float, float, float]


def load_pelvis_calibration(
    source: Path | Mapping[str, object],
) -> PelvisReferenceCalibration:
    if isinstance(source, Path):
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OptiTrackReferenceError(
                f"{source}: invalid calibration JSON"
            ) from exc
    else:
        payload = dict(source)
    if not isinstance(payload, Mapping):
        raise OptiTrackReferenceError("calibration must be an object")
    if payload.get("schema") != CALIBRATION_SCHEMA:
        raise OptiTrackReferenceError("unsupported calibration schema")
    if payload.get("status") != "validated":
        raise OptiTrackReferenceError(
            "pelvis calibration status must be validated"
        )
    if payload.get("tracked_link") != "pelvis":
        raise OptiTrackReferenceError(
            "this converter requires a direct pelvis rigid body; a head or "
            "torso marker requires synchronized joint FK and a separate "
            "derived-reference contract"
        )

    motive_translation, motive_quaternion = _require_transform(
        payload,
        "T_target_motive_world",
    )
    pelvis_translation, pelvis_quaternion = _require_transform(
        payload,
        "T_rigid_body_pelvis",
    )
    try:
        calibration = PelvisReferenceCalibration(
            calibration_id=str(payload["calibration_id"]),
            raw_capture_calibration_id=str(
                payload["raw_capture_calibration_id"]
            ),
            rigid_body_id=int(payload["rigid_body_id"]),
            rigid_body_name=str(payload["rigid_body_name"]),
            target_frame_id=str(payload["target_frame_id"]),
            clock_id=str(payload["clock_id"]),
            clock_mapping_method=str(payload["clock_mapping_method"]),
            clock_mapping_calibration_id=str(
                payload["clock_mapping_calibration_id"]
            ),
            clock_mapping_residual_p95_ns=int(
                payload["clock_mapping_residual_p95_ns"]
            ),
            max_clock_mapping_residual_ns=int(
                payload["max_clock_mapping_residual_ns"]
            ),
            maximum_marker_error_m=float(
                payload["maximum_marker_error_m"]
            ),
            maximum_velocity_interval_ns=int(
                payload["maximum_velocity_interval_ns"]
            ),
            motive_world_to_target_translation_m=motive_translation,
            motive_world_to_target_quaternion_wxyz=motive_quaternion,
            rigid_body_to_pelvis_translation_m=pelvis_translation,
            rigid_body_to_pelvis_quaternion_wxyz=pelvis_quaternion,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise OptiTrackReferenceError(
            "calibration has missing or invalid fields"
        ) from exc

    for field_name in (
        "calibration_id",
        "raw_capture_calibration_id",
        "rigid_body_name",
        "target_frame_id",
        "clock_id",
        "clock_mapping_method",
        "clock_mapping_calibration_id",
    ):
        if not getattr(calibration, field_name):
            raise OptiTrackReferenceError(f"{field_name} must be non-empty")
    if calibration.rigid_body_id < 0:
        raise OptiTrackReferenceError("rigid_body_id must be non-negative")
    if (
        calibration.clock_mapping_residual_p95_ns < 0
        or calibration.max_clock_mapping_residual_ns < 0
        or calibration.clock_mapping_residual_p95_ns
        > calibration.max_clock_mapping_residual_ns
    ):
        raise OptiTrackReferenceError(
            "clock mapping residual exceeds the frozen clock limit"
        )
    if (
        not math.isfinite(calibration.maximum_marker_error_m)
        or calibration.maximum_marker_error_m <= 0.0
    ):
        raise OptiTrackReferenceError(
            "maximum_marker_error_m must be finite and positive"
        )
    if calibration.maximum_velocity_interval_ns <= 0:
        raise OptiTrackReferenceError(
            "maximum_velocity_interval_ns must be positive"
        )
    return calibration


def _rotation_from_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _wxyz_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        root = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            0.25 * root,
            (rotation[2, 1] - rotation[1, 2]) / root,
            (rotation[0, 2] - rotation[2, 0]) / root,
            (rotation[1, 0] - rotation[0, 1]) / root,
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            root = math.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            ) * 2.0
            quaternion = (
                (rotation[2, 1] - rotation[1, 2]) / root,
                0.25 * root,
                (rotation[0, 1] + rotation[1, 0]) / root,
                (rotation[0, 2] + rotation[2, 0]) / root,
            )
        elif index == 1:
            root = math.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            ) * 2.0
            quaternion = (
                (rotation[0, 2] - rotation[2, 0]) / root,
                (rotation[0, 1] + rotation[1, 0]) / root,
                0.25 * root,
                (rotation[1, 2] + rotation[2, 1]) / root,
            )
        else:
            root = math.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            ) * 2.0
            quaternion = (
                (rotation[1, 0] - rotation[0, 1]) / root,
                (rotation[0, 2] + rotation[2, 0]) / root,
                (rotation[1, 2] + rotation[2, 1]) / root,
                0.25 * root,
            )
    values = np.asarray(quaternion, dtype=np.float64)
    values /= np.linalg.norm(values)
    if values[0] < 0.0:
        values = -values
    return tuple(float(value) for value in values)


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) / 2.0))
    angle = math.acos(cosine)
    if angle < 1e-10:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * math.sin(angle)
    return axis * angle


@dataclass
class _ConvertedPose:
    output: dict[str, object]
    valid: bool
    time_ns: int
    position: np.ndarray | None
    rotation: np.ndarray | None


def _metadata_record(
    raw_metadata: Mapping[str, object],
    calibration: PelvisReferenceCalibration,
    calibration_sha256: str,
) -> dict[str, object]:
    return {
        "schema": REFERENCE_SCHEMA,
        "kind": "metadata",
        "pose_semantics": "pelvis_pose",
        "clock_id": calibration.clock_id,
        "frame_id": calibration.target_frame_id,
        "calibration_id": calibration.calibration_id,
        "calibration_sha256": calibration_sha256,
        "source": "optitrack_natnet_direct_pelvis",
        "raw_schema": RAW_SCHEMA,
        "raw_natnet_sdk_archive_sha256": raw_metadata.get(
            "natnet_sdk_archive_sha256"
        ),
        "raw_capture_calibration_id": calibration.raw_capture_calibration_id,
        "clock_mapping_method": calibration.clock_mapping_method,
        "clock_mapping_calibration_id": (
            calibration.clock_mapping_calibration_id
        ),
        "clock_mapping_residual_p95_ns": (
            calibration.clock_mapping_residual_p95_ns
        ),
        "maximum_marker_error_m": calibration.maximum_marker_error_m,
        "velocity_method": "adjacent_valid_pose_finite_difference",
        "transform_convention": (
            "T_A_B maps coordinates from frame B into frame A"
        ),
    }


def _invalid_reason(
    record: Mapping[str, object],
    calibration: PelvisReferenceCalibration,
) -> str | None:
    if record.get("rigid_body_present") is not True:
        return "rigid_body_missing"
    if record.get("tracking_valid") is not True:
        return "tracking_invalid"
    marker_error = record.get("mean_marker_error_m")
    if not isinstance(marker_error, (int, float)) or not math.isfinite(
        float(marker_error)
    ):
        return "marker_error_missing"
    if float(marker_error) > calibration.maximum_marker_error_m:
        return "marker_error_exceeds_limit"
    if record.get("capture_realtime_estimate_ns") is None:
        return "capture_time_unavailable"
    if record.get("position_motive_xyz_m") is None:
        return "position_unavailable"
    if record.get("quaternion_motive_xyzw") is None:
        return "orientation_unavailable"
    return None


def _convert_pose(
    record: Mapping[str, object],
    calibration: PelvisReferenceCalibration,
    target_from_motive_rotation: np.ndarray,
    target_from_motive_translation: np.ndarray,
    rigid_from_pelvis_rotation: np.ndarray,
    rigid_from_pelvis_translation: np.ndarray,
) -> _ConvertedPose:
    try:
        sequence = int(record["frame_number"])
        receipt_time_ns = int(record["receipt_realtime_ns"])
        capture_value = record.get("capture_realtime_estimate_ns")
        time_ns = receipt_time_ns if capture_value is None else int(capture_value)
    except (KeyError, TypeError, ValueError) as exc:
        raise OptiTrackReferenceError("raw pose has invalid timing") from exc

    reason = _invalid_reason(record, calibration)
    base_output: dict[str, object] = {
        "schema": REFERENCE_SCHEMA,
        "kind": "reference_pose",
        "sequence": sequence,
        "time_ns": time_ns,
        "tracking_valid": reason is None,
        "mean_marker_error_m": record.get("mean_marker_error_m"),
        "raw_frame_number": sequence,
        "raw_receipt_realtime_ns": receipt_time_ns,
        "raw_motive_software_time_s": record.get("motive_software_time_s"),
    }
    if reason is not None:
        base_output.update(
            {
                "invalid_reason": reason,
                "position_m": None,
                "quaternion_wxyz": None,
            }
        )
        return _ConvertedPose(base_output, False, time_ns, None, None)

    try:
        raw_position = np.asarray(
            _tuple_of_floats(
                record["position_motive_xyz_m"],
                expected_length=3,
                field_name="position_motive_xyz_m",
            ),
            dtype=np.float64,
        )
        raw_xyzw = _tuple_of_floats(
            record["quaternion_motive_xyzw"],
            expected_length=4,
            field_name="quaternion_motive_xyzw",
        )
    except KeyError as exc:
        raise OptiTrackReferenceError("raw pose is missing geometry") from exc
    raw_rotation = _rotation_from_wxyz(
        (raw_xyzw[3], raw_xyzw[0], raw_xyzw[1], raw_xyzw[2])
    )
    target_rotation = (
        target_from_motive_rotation
        @ raw_rotation
        @ rigid_from_pelvis_rotation
    )
    target_position = (
        target_from_motive_translation
        + target_from_motive_rotation
        @ (raw_position + raw_rotation @ rigid_from_pelvis_translation)
    )
    base_output.update(
        {
            "position_m": [float(value) for value in target_position],
            "quaternion_wxyz": list(_wxyz_from_rotation(target_rotation)),
        }
    )
    return _ConvertedPose(
        base_output,
        True,
        time_ns,
        target_position,
        target_rotation,
    )


def _add_velocities(
    converted: list[_ConvertedPose],
    *,
    maximum_interval_ns: int,
) -> None:
    for index, current in enumerate(converted):
        if not current.valid:
            continue
        neighbor_pairs: list[tuple[_ConvertedPose, _ConvertedPose]] = []
        if (
            index > 0
            and index + 1 < len(converted)
            and converted[index - 1].valid
            and converted[index + 1].valid
            and (
                converted[index + 1].time_ns
                - converted[index - 1].time_ns
                <= maximum_interval_ns
            )
        ):
            neighbor_pairs.append((converted[index - 1], converted[index + 1]))
        elif index + 1 < len(converted) and converted[index + 1].valid:
            neighbor_pairs.append((current, converted[index + 1]))
        elif index > 0 and converted[index - 1].valid:
            neighbor_pairs.append((converted[index - 1], current))
        if not neighbor_pairs:
            continue
        first, second = neighbor_pairs[0]
        interval_ns = second.time_ns - first.time_ns
        if interval_ns <= 0 or interval_ns > maximum_interval_ns:
            continue
        assert first.position is not None and second.position is not None
        assert first.rotation is not None and second.rotation is not None
        interval_s = interval_ns / 1.0e9
        linear_velocity = (second.position - first.position) / interval_s
        world_rotation_delta = second.rotation @ first.rotation.T
        angular_velocity = _rotation_vector(world_rotation_delta) / interval_s
        current.output["linear_velocity_mps"] = [
            float(value) for value in linear_velocity
        ]
        current.output["angular_velocity_radps"] = [
            float(value) for value in angular_velocity
        ]


def convert_raw_optitrack_records(
    records: Sequence[Mapping[str, object]],
    *,
    calibration: PelvisReferenceCalibration,
    calibration_sha256: str,
) -> list[dict[str, object]]:
    if len(calibration_sha256) != 64:
        raise OptiTrackReferenceError(
            "calibration_sha256 must contain 64 hexadecimal characters"
        )
    metadata_records = [
        record for record in records if record.get("kind") == "metadata"
    ]
    if len(metadata_records) != 1:
        raise OptiTrackReferenceError(
            "raw capture must contain exactly one metadata record"
        )
    metadata = metadata_records[0]
    if (
        metadata.get("schema") != RAW_SCHEMA
        or metadata.get("pose_semantics") != "rigid_body_pose_raw"
    ):
        raise OptiTrackReferenceError("unsupported raw capture contract")
    expected_identity = (
        calibration.rigid_body_id,
        calibration.rigid_body_name,
        calibration.raw_capture_calibration_id,
    )
    observed_identity = (
        metadata.get("rigid_body_id"),
        metadata.get("rigid_body_name"),
        metadata.get("calibration_id"),
    )
    if observed_identity != expected_identity:
        raise OptiTrackReferenceError(
            "raw rigid-body identity does not match calibration identity"
        )

    target_from_motive_translation = np.asarray(
        calibration.motive_world_to_target_translation_m,
        dtype=np.float64,
    )
    target_from_motive_rotation = _rotation_from_wxyz(
        calibration.motive_world_to_target_quaternion_wxyz
    )
    rigid_from_pelvis_translation = np.asarray(
        calibration.rigid_body_to_pelvis_translation_m,
        dtype=np.float64,
    )
    rigid_from_pelvis_rotation = _rotation_from_wxyz(
        calibration.rigid_body_to_pelvis_quaternion_wxyz
    )

    raw_poses = [
        record for record in records if record.get("kind") == "raw_rigid_body_pose"
    ]
    if not raw_poses:
        raise OptiTrackReferenceError("raw capture contains no pose records")
    frame_numbers = [int(record["frame_number"]) for record in raw_poses]
    if not all(
        previous < current
        for previous, current in zip(frame_numbers, frame_numbers[1:])
    ):
        raise OptiTrackReferenceError(
            "raw frame numbers must be strictly increasing"
        )
    for record in raw_poses:
        if record.get("schema") != RAW_SCHEMA:
            raise OptiTrackReferenceError("raw pose schema mismatch")
        if record.get("rigid_body_id") != calibration.rigid_body_id:
            raise OptiTrackReferenceError(
                "raw sample rigid-body identity mismatch"
            )

    converted = [
        _convert_pose(
            record,
            calibration,
            target_from_motive_rotation,
            target_from_motive_translation,
            rigid_from_pelvis_rotation,
            rigid_from_pelvis_translation,
        )
        for record in raw_poses
    ]
    if not all(
        previous.time_ns < current.time_ns
        for previous, current in zip(converted, converted[1:])
    ):
        raise OptiTrackReferenceError(
            "normalized capture times must be strictly increasing"
        )
    _add_velocities(
        converted,
        maximum_interval_ns=calibration.maximum_velocity_interval_ns,
    )
    return [
        _metadata_record(metadata, calibration, calibration_sha256),
        *(item.output for item in converted),
    ]
