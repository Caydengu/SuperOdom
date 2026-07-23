"""Versioned JSONL input/output for independent dynamic-reference scoring."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from g1_root_state_bridge.dynamic_reference import (
    ClockMapping,
    DynamicReferenceConfig,
    DynamicReferenceReport,
    PoseSample,
    WaistJointSample,
)
from g1_root_state_bridge.joint_contract import CANONICAL_G1_JOINT_NAMES
from g1_root_state_bridge.joint_transport import (
    canonical_joint_mapping_digest,
)
from g1_root_state_bridge.protocol import (
    RootStateProtocolError,
    deserialize_root_state_v2,
)


REFERENCE_SCHEMA = "g1_dynamic_reference_v1"
SCORE_CONFIG_SCHEMA = "g1_dynamic_reference_score_config_v1"
SCORE_REPORT_SCHEMA = "g1_dynamic_reference_score_v1"


class DynamicReferenceFormatError(ValueError):
    """Raised when an input would make dynamic evidence ambiguous."""


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DynamicReferenceFormatError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(record, dict):
                raise DynamicReferenceFormatError(
                    f"{path}:{line_number}: record must be an object"
                )
            records.append(record)
    return records


def load_estimator_status_jsonl(
    path: Path,
    *,
    clock_id: str,
    frame_id: str,
) -> list[PoseSample]:
    """Load typed V2 packets from the bridge status trace.

    Duplicated human-readable fields in the JSONL are deliberately ignored;
    the serialized V2 payload remains the source of truth.
    """

    samples: list[PoseSample] = []
    for record_number, record in enumerate(_read_jsonl(path), start=1):
        if record.get("kind") != "packet":
            continue
        payload_hex = record.get("payload_hex")
        if not isinstance(payload_hex, str):
            raise DynamicReferenceFormatError(
                f"{path}: record {record_number}: packet lacks payload_hex"
            )
        try:
            packet = deserialize_root_state_v2(bytes.fromhex(payload_hex))
        except (ValueError, RootStateProtocolError) as exc:
            raise DynamicReferenceFormatError(
                f"{path}: record {record_number}: invalid V2 payload"
            ) from exc
        record_valid = record.get("strictly_valid", packet.strictly_valid)
        if not isinstance(record_valid, bool):
            raise DynamicReferenceFormatError(
                f"{path}: record {record_number}: strictly_valid must be bool"
            )
        samples.append(
            PoseSample(
                sequence=packet.sequence,
                time_ns=packet.estimate_time_ns,
                clock_id=clock_id,
                frame_id=frame_id,
                calibration_id=packet.calibration_digest.hex(),
                position=packet.position,
                quaternion_wxyz=packet.quaternion_wxyz,
                linear_velocity=packet.linear_velocity,
                angular_velocity=packet.angular_velocity,
                valid=packet.strictly_valid and record_valid,
            )
        )
    if not samples:
        raise DynamicReferenceFormatError(
            f"{path}: no serialized V2 packet records found"
        )
    return samples


def load_waist_joint_evidence_jsonl(path: Path) -> list[WaistJointSample]:
    """Load the synchronized joints used by FK from each bridge packet record."""

    samples: list[WaistJointSample] = []
    expected_names = list(CANONICAL_G1_JOINT_NAMES)
    expected_mapping = canonical_joint_mapping_digest().hex()
    for record_number, record in enumerate(_read_jsonl(path), start=1):
        if record.get("kind") != "packet":
            continue
        payload_hex = record.get("payload_hex")
        if not isinstance(payload_hex, str):
            raise DynamicReferenceFormatError(
                f"{path}: record {record_number}: packet lacks payload_hex"
            )
        try:
            packet = deserialize_root_state_v2(bytes.fromhex(payload_hex))
        except (ValueError, RootStateProtocolError) as exc:
            raise DynamicReferenceFormatError(
                f"{path}: record {record_number}: invalid V2 payload"
            ) from exc

        if record.get("joint_mapping_digest_sha256") != expected_mapping:
            raise DynamicReferenceFormatError(
                f"{path}: record {record_number}: joint mapping is not the "
                "canonical G1 contract"
            )
        if record.get("joint_names") != expected_names:
            raise DynamicReferenceFormatError(
                f"{path}: record {record_number}: joint names are not in "
                "canonical G1 order"
            )
        raw_position = record.get("joint_position")
        raw_velocity = record.get("joint_velocity")
        if (
            not isinstance(raw_position, list)
            or len(raw_position) != len(CANONICAL_G1_JOINT_NAMES)
            or not isinstance(raw_velocity, list)
            or len(raw_velocity) != len(CANONICAL_G1_JOINT_NAMES)
        ):
            raise DynamicReferenceFormatError(
                f"{path}: record {record_number}: synchronized joint evidence "
                "must contain 29 positions and velocities"
            )
        try:
            position = tuple(float(value) for value in raw_position)
            velocity = tuple(float(value) for value in raw_velocity)
        except (TypeError, ValueError) as exc:
            raise DynamicReferenceFormatError(
                f"{path}: record {record_number}: joint evidence is malformed"
            ) from exc
        if not all(math.isfinite(value) for value in (*position, *velocity)):
            raise DynamicReferenceFormatError(
                f"{path}: record {record_number}: joint evidence must be finite"
            )
        record_valid = record.get("strictly_valid", packet.strictly_valid)
        if not isinstance(record_valid, bool):
            raise DynamicReferenceFormatError(
                f"{path}: record {record_number}: strictly_valid must be bool"
            )
        samples.append(
            WaistJointSample(
                sequence=packet.sequence,
                time_ns=packet.estimate_time_ns,
                position_rad=(
                    position[12],
                    position[13],
                    position[14],
                ),
                valid=packet.strictly_valid and record_valid,
            )
        )
    if not samples:
        raise DynamicReferenceFormatError(
            f"{path}: no synchronized waist-joint packet evidence found"
        )
    return samples


def _require_string(
    record: Mapping[str, object],
    field_name: str,
    *,
    context: str,
) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value:
        raise DynamicReferenceFormatError(
            f"{context}: {field_name} must be a non-empty string"
        )
    return value


def _optional_vector(
    record: Mapping[str, object],
    field_name: str,
) -> tuple[float, float, float] | None:
    value = record.get(field_name)
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise DynamicReferenceFormatError(f"{field_name} must be a vector")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def load_reference_pose_jsonl(path: Path) -> list[PoseSample]:
    """Load normalized pelvis truth, never an unconverted rigid-link pose."""

    records = _read_jsonl(path)
    metadata_records = [
        record for record in records if record.get("kind") == "metadata"
    ]
    if len(metadata_records) != 1:
        raise DynamicReferenceFormatError(
            f"{path}: expected exactly one metadata record"
        )
    metadata = metadata_records[0]
    if metadata.get("schema") != REFERENCE_SCHEMA:
        raise DynamicReferenceFormatError(
            f"{path}: unsupported reference schema"
        )
    if metadata.get("pose_semantics") != "pelvis_pose":
        raise DynamicReferenceFormatError(
            f"{path}: pose_semantics must be pelvis_pose; raw link poses "
            "require an explicit calibrated conversion"
        )
    clock_id = _require_string(metadata, "clock_id", context=str(path))
    frame_id = _require_string(metadata, "frame_id", context=str(path))
    calibration_id = _require_string(
        metadata,
        "calibration_id",
        context=str(path),
    )

    samples: list[PoseSample] = []
    for record_number, record in enumerate(records, start=1):
        if record.get("kind") != "reference_pose":
            continue
        if record.get("schema") != REFERENCE_SCHEMA:
            raise DynamicReferenceFormatError(
                f"{path}: record {record_number}: schema mismatch"
            )
        try:
            sequence = int(record["sequence"])
            time_ns = int(record["time_ns"])
            tracking_valid = record["tracking_valid"]
        except (KeyError, TypeError, ValueError) as exc:
            raise DynamicReferenceFormatError(
                f"{path}: record {record_number}: malformed reference pose"
            ) from exc
        if not isinstance(tracking_valid, bool):
            raise DynamicReferenceFormatError(
                f"{path}: record {record_number}: tracking_valid must be bool"
            )
        raw_position = record.get("position_m")
        raw_quaternion = record.get("quaternion_wxyz")
        if not tracking_valid and (
            raw_position is None or raw_quaternion is None
        ):
            position = (0.0, 0.0, 0.0)
            quaternion = (1.0, 0.0, 0.0, 0.0)
        else:
            try:
                position = tuple(float(item) for item in raw_position)
                quaternion = tuple(float(item) for item in raw_quaternion)
            except (TypeError, ValueError) as exc:
                raise DynamicReferenceFormatError(
                    f"{path}: record {record_number}: malformed pose geometry"
                ) from exc
        marker_error = record.get("mean_marker_error_m")
        if marker_error is not None:
            try:
                marker_error = float(marker_error)
            except (TypeError, ValueError) as exc:
                raise DynamicReferenceFormatError(
                    f"{path}: record {record_number}: invalid marker error"
                ) from exc
        try:
            sample = PoseSample(
                sequence=sequence,
                time_ns=time_ns,
                clock_id=clock_id,
                frame_id=frame_id,
                calibration_id=calibration_id,
                position=position,  # type: ignore[arg-type]
                quaternion_wxyz=quaternion,  # type: ignore[arg-type]
                linear_velocity=_optional_vector(
                    record,
                    "linear_velocity_mps",
                ),
                angular_velocity=_optional_vector(
                    record,
                    "angular_velocity_radps",
                ),
                valid=tracking_valid,
                tracking_error_m=marker_error,
            )
        except ValueError as exc:
            raise DynamicReferenceFormatError(
                f"{path}: record {record_number}: {exc}"
            ) from exc
        samples.append(sample)
    if not samples:
        raise DynamicReferenceFormatError(
            f"{path}: no normalized pelvis reference poses found"
        )
    return samples


def load_score_config_json(
    path: Path,
) -> tuple[DynamicReferenceConfig, ClockMapping | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DynamicReferenceFormatError(f"{path}: invalid config JSON") from exc
    if not isinstance(payload, dict):
        raise DynamicReferenceFormatError(f"{path}: config must be an object")
    if payload.get("schema") != SCORE_CONFIG_SCHEMA:
        raise DynamicReferenceFormatError(f"{path}: unsupported config schema")

    config_fields = (
        "target_clock_id",
        "reference_calibration_id",
        "calibration_interval_ns",
        "evaluation_interval_ns",
        "waist_interval_ns",
        "max_interpolation_gap_ns",
        "max_clock_mapping_residual_ns",
        "minimum_coverage",
        "minimum_velocity_coverage",
        "horizontal_p95_limit_m",
        "vertical_p95_limit_m",
        "yaw_p95_limit_deg",
        "waist_position_residual_limit_m",
        "waist_yaw_residual_limit_deg",
        "minimum_horizontal_excitation_m",
        "minimum_vertical_excitation_m",
        "minimum_yaw_excitation_deg",
        "minimum_waist_joint_excitation_rad",
        "maximum_waist_reference_translation_m",
        "maximum_waist_reference_yaw_deg",
    )
    try:
        config_values = {field: payload[field] for field in config_fields}
        for interval_field in (
            "calibration_interval_ns",
            "evaluation_interval_ns",
            "waist_interval_ns",
        ):
            config_values[interval_field] = tuple(config_values[interval_field])
        config = DynamicReferenceConfig(**config_values)
    except (KeyError, TypeError, ValueError) as exc:
        raise DynamicReferenceFormatError(
            f"{path}: invalid score configuration"
        ) from exc

    mapping_payload = payload.get("clock_mapping")
    if mapping_payload is None:
        return config, None
    if not isinstance(mapping_payload, dict):
        raise DynamicReferenceFormatError(
            f"{path}: clock_mapping must be an object"
        )
    try:
        mapping = ClockMapping(
            source_clock_id=mapping_payload["source_clock_id"],
            target_clock_id=mapping_payload["target_clock_id"],
            scale=mapping_payload["scale"],
            offset_ns=mapping_payload["offset_ns"],
            residual_p95_ns=mapping_payload["residual_p95_ns"],
            calibration_id=mapping_payload["calibration_id"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DynamicReferenceFormatError(
            f"{path}: invalid clock_mapping"
        ) from exc
    return config, mapping


def report_as_dict(report: DynamicReferenceReport) -> dict[str, object]:
    return {
        "schema": SCORE_REPORT_SCHEMA,
        "valid": report.valid,
        "gates_pass": report.gates_pass,
        "invalid_reasons": list(report.invalid_reasons),
        "metrics": dict(report.metrics),
        "gate_results": dict(report.gate_results),
        "alignment_rotation": report.alignment_rotation,
        "alignment_translation": report.alignment_translation,
        "alignment_scale": report.alignment_scale,
    }
