from __future__ import annotations

import json
import math

import pytest

from g1_root_state_bridge.dynamic_reference_io import (
    DynamicReferenceFormatError,
    load_estimator_status_jsonl,
    load_reference_pose_jsonl,
    load_score_config_json,
    load_waist_joint_evidence_jsonl,
    report_as_dict,
)
from g1_root_state_bridge.dynamic_reference_cli import main as score_main
from g1_root_state_bridge.joint_contract import CANONICAL_G1_JOINT_NAMES
from g1_root_state_bridge.joint_transport import (
    canonical_joint_mapping_digest,
)
from g1_root_state_bridge.protocol import (
    REQUIRED_HEALTH_FLAGS,
    RootStatePacketV2,
    serialize_root_state_v2,
)
from g1_root_state_bridge.dynamic_reference import score_dynamic_reference


def _write_jsonl(path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _packet(sequence: int, time_ns: int) -> RootStatePacketV2:
    if sequence <= 3:
        x, z, yaw = 0.0, 0.8, 0.0
    elif sequence <= 6:
        fraction = (sequence - 4) / 2.0
        x, z, yaw = 0.2 * fraction, 0.8 + 0.08 * fraction, 0.1 * fraction
    else:
        x, z, yaw = 0.2, 0.88, 0.1
    return RootStatePacketV2(
        sequence=sequence,
        source_epoch=17,
        estimate_time_ns=time_ns,
        publish_time_ns=time_ns + 1_000_000,
        correction_time_ns=time_ns - 50_000_000,
        joint_time_ns=time_ns - 100_000,
        joint_sync_gap_ns=-100_000,
        health_flags=REQUIRED_HEALTH_FLAGS,
        position=(x, 0.0, z),
        quaternion_wxyz=(
            math.cos(yaw / 2.0),
            0.0,
            0.0,
            math.sin(yaw / 2.0),
        ),
        linear_velocity=(0.1, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        covariance_diagonal=(1.0,) * 6,
        calibration_digest=bytes(range(32)),
    )


def test_root_state_status_loader_uses_serialized_v2_as_source_of_truth(
    tmp_path,
) -> None:
    first = _packet(1, 1_000_000_000)
    second = _packet(2, 1_100_000_000)
    path = tmp_path / "bridge_status.jsonl"
    _write_jsonl(
        path,
        [
            {"schema": "g1_root_state_bridge_status_v1", "event": "startup"},
            {
                "schema": "g1_root_state_bridge_status_v1",
                "event": "packet_published",
                "kind": "packet",
                "strictly_valid": True,
                "payload_hex": serialize_root_state_v2(first).hex(),
                "position": [999.0, 999.0, 999.0],
            },
            {
                "schema": "g1_root_state_bridge_status_v1",
                "event": "packet_published",
                "kind": "packet",
                "strictly_valid": True,
                "payload_hex": serialize_root_state_v2(second).hex(),
            },
        ],
    )

    samples = load_estimator_status_jsonl(
        path,
        clock_id="unix_realtime_ns",
        frame_id="superodom_world",
    )

    assert len(samples) == 2
    assert samples[0].position == pytest.approx(first.position)
    assert samples[0].position != (999.0, 999.0, 999.0)
    assert samples[0].calibration_id == first.calibration_digest.hex()
    assert samples[1].sequence == 2


def test_waist_joint_loader_requires_hash_bound_synchronized_joint_evidence(
    tmp_path,
) -> None:
    packet = _packet(7, 1_600_000_000)
    path = tmp_path / "bridge_status.jsonl"
    _write_jsonl(
        path,
        [
            {
                "schema": "g1_root_state_bridge_status_v1",
                "event": "packet_published",
                "kind": "packet",
                "strictly_valid": True,
                "payload_hex": serialize_root_state_v2(packet).hex(),
                "joint_names": list(CANONICAL_G1_JOINT_NAMES),
                "joint_position": [0.0] * 12
                + [0.2, -0.1, 0.15]
                + [0.0] * 14,
                "joint_velocity": [0.0] * 29,
                "joint_mapping_digest_sha256": (
                    canonical_joint_mapping_digest().hex()
                ),
            }
        ],
    )

    samples = load_waist_joint_evidence_jsonl(path)

    assert len(samples) == 1
    assert samples[0].time_ns == packet.estimate_time_ns
    assert samples[0].position_rad == pytest.approx((0.2, -0.1, 0.15))

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    records[0]["joint_mapping_digest_sha256"] = "0" * 64
    _write_jsonl(path, records)
    with pytest.raises(DynamicReferenceFormatError, match="mapping"):
        load_waist_joint_evidence_jsonl(path)


def test_normalized_reference_contract_preserves_identity_and_quality(
    tmp_path,
) -> None:
    path = tmp_path / "reference.jsonl"
    _write_jsonl(
        path,
        [
            {
                "schema": "g1_dynamic_reference_v1",
                "kind": "metadata",
                "pose_semantics": "pelvis_pose",
                "clock_id": "motive_software_ns",
                "frame_id": "motive_world_y_up",
                "calibration_id": "pelvis-rigid-body-2026-07-22",
                "source": "optitrack_natnet",
            },
            {
                "schema": "g1_dynamic_reference_v1",
                "kind": "reference_pose",
                "sequence": 8801,
                "time_ns": 25_000_000_000,
                "position_m": [1.0, 2.0, 3.0],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "linear_velocity_mps": [0.1, 0.0, 0.0],
                "angular_velocity_radps": [0.0, 0.0, 0.2],
                "tracking_valid": True,
                "mean_marker_error_m": 0.0007,
            },
            {
                "schema": "g1_dynamic_reference_v1",
                "kind": "reference_pose",
                "sequence": 8802,
                "time_ns": 25_008_333_333,
                "position_m": [1.001, 2.0, 3.0],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "tracking_valid": False,
                "mean_marker_error_m": 0.002,
            },
        ],
    )

    samples = load_reference_pose_jsonl(path)

    assert len(samples) == 2
    assert samples[0].sequence == 8801
    assert samples[0].clock_id == "motive_software_ns"
    assert samples[0].frame_id == "motive_world_y_up"
    assert samples[0].calibration_id == "pelvis-rigid-body-2026-07-22"
    assert samples[0].tracking_error_m == pytest.approx(0.0007)
    assert samples[1].valid is False


def test_reference_loader_rejects_raw_link_pose_as_pelvis_truth(tmp_path) -> None:
    path = tmp_path / "raw_reference.jsonl"
    _write_jsonl(
        path,
        [
            {
                "schema": "g1_dynamic_reference_v1",
                "kind": "metadata",
                "pose_semantics": "head_rigid_body_raw",
                "clock_id": "motive_software_ns",
                "frame_id": "motive_world_y_up",
                "calibration_id": "uncalibrated-head",
                "source": "optitrack_natnet",
            }
        ],
    )

    with pytest.raises(DynamicReferenceFormatError, match="pelvis_pose"):
        load_reference_pose_jsonl(path)


def test_score_config_loads_clock_mapping_and_serializes_report(tmp_path) -> None:
    path = tmp_path / "score_config.json"
    path.write_text(
        json.dumps(
            {
                "schema": "g1_dynamic_reference_score_config_v1",
                "target_clock_id": "unix_realtime_ns",
                "reference_calibration_id": "pelvis-rigid-body-2026-07-22",
                "calibration_interval_ns": [1_000_000_000, 2_000_000_000],
                "evaluation_interval_ns": [2_000_000_001, 4_000_000_000],
                "waist_interval_ns": [4_000_000_001, 6_000_000_000],
                "max_interpolation_gap_ns": 20_000_000,
                "max_clock_mapping_residual_ns": 5_000_000,
                "minimum_coverage": 0.9,
                "minimum_velocity_coverage": 0.9,
                "horizontal_p95_limit_m": 0.05,
                "vertical_p95_limit_m": 0.03,
                "yaw_p95_limit_deg": 1.5,
                "waist_position_residual_limit_m": 0.02,
                "waist_yaw_residual_limit_deg": 0.5,
                "minimum_horizontal_excitation_m": 0.10,
                "minimum_vertical_excitation_m": 0.06,
                "minimum_yaw_excitation_deg": 3.0,
                "minimum_waist_joint_excitation_rad": 0.0872664626,
                "maximum_waist_reference_translation_m": 0.02,
                "maximum_waist_reference_yaw_deg": 0.5,
                "clock_mapping": {
                    "source_clock_id": "motive_software_ns",
                    "target_clock_id": "unix_realtime_ns",
                    "scale": 1.0,
                    "offset_ns": -24_000_000_000,
                    "residual_p95_ns": 1_000_000,
                    "calibration_id": "motive-to-oslo-v1",
                },
            }
        ),
        encoding="utf-8",
    )

    config, mapping = load_score_config_json(path)

    assert config.horizontal_p95_limit_m == pytest.approx(0.05)
    assert mapping is not None
    assert mapping.calibration_id == "motive-to-oslo-v1"

    invalid = score_dynamic_reference([], [], config=config, clock_mapping=mapping)
    serialized = report_as_dict(invalid)
    assert serialized["schema"] == "g1_dynamic_reference_score_v1"
    assert serialized["valid"] is False
    assert serialized["gates_pass"] is False
    assert "insufficient_estimator_samples" in serialized["invalid_reasons"]


def test_score_cli_writes_hash_bound_pass_report(tmp_path) -> None:
    estimator_path = tmp_path / "estimator.jsonl"
    reference_path = tmp_path / "reference.jsonl"
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "score.json"

    estimator_records: list[dict[str, object]] = []
    reference_records: list[dict[str, object]] = [
        {
            "schema": "g1_dynamic_reference_v1",
            "kind": "metadata",
            "pose_semantics": "pelvis_pose",
            "clock_id": "unix_realtime_ns",
            "frame_id": "motive_world",
            "calibration_id": "direct-pelvis-rigid-body-v1",
            "source": "synthetic-test",
        }
    ]
    for index in range(9):
        time_ns = 1_000_000_000 + index * 100_000_000
        packet = _packet(index + 1, time_ns)
        estimator_records.append(
            {
                "schema": "g1_root_state_bridge_status_v1",
                "event": "packet_published",
                "kind": "packet",
                "strictly_valid": True,
                "payload_hex": serialize_root_state_v2(packet).hex(),
                "joint_names": list(CANONICAL_G1_JOINT_NAMES),
                "joint_position": [0.0] * 12
                + [
                    0.1 * max(0, index - 6),
                    0.0,
                    0.0,
                ]
                + [0.0] * 14,
                "joint_velocity": [0.0] * 29,
                "joint_mapping_digest_sha256": (
                    canonical_joint_mapping_digest().hex()
                ),
            }
        )
        reference_records.append(
            {
                "schema": "g1_dynamic_reference_v1",
                "kind": "reference_pose",
                "sequence": index + 1,
                "time_ns": time_ns,
                "position_m": list(packet.position),
                "quaternion_wxyz": list(packet.quaternion_wxyz),
                "linear_velocity_mps": list(packet.linear_velocity),
                "angular_velocity_radps": list(packet.angular_velocity),
                "tracking_valid": True,
                "mean_marker_error_m": 0.0005,
            }
        )
    _write_jsonl(estimator_path, estimator_records)
    _write_jsonl(reference_path, reference_records)
    config_path.write_text(
        json.dumps(
            {
                "schema": "g1_dynamic_reference_score_config_v1",
                "target_clock_id": "unix_realtime_ns",
                "reference_calibration_id": "direct-pelvis-rigid-body-v1",
                "calibration_interval_ns": [1_000_000_000, 1_200_000_000],
                "evaluation_interval_ns": [1_300_000_000, 1_500_000_000],
                "waist_interval_ns": [1_600_000_000, 1_800_000_000],
                "max_interpolation_gap_ns": 150_000_000,
                "max_clock_mapping_residual_ns": 5_000_000,
                "minimum_coverage": 1.0,
                "minimum_velocity_coverage": 1.0,
                "horizontal_p95_limit_m": 0.05,
                "vertical_p95_limit_m": 0.03,
                "yaw_p95_limit_deg": 1.5,
                "waist_position_residual_limit_m": 0.02,
                "waist_yaw_residual_limit_deg": 0.5,
                "minimum_horizontal_excitation_m": 0.10,
                "minimum_vertical_excitation_m": 0.06,
                "minimum_yaw_excitation_deg": 3.0,
                "minimum_waist_joint_excitation_rad": 0.0872664626,
                "maximum_waist_reference_translation_m": 0.02,
                "maximum_waist_reference_yaw_deg": 0.5,
            }
        ),
        encoding="utf-8",
    )

    return_code = score_main(
        [
            "--estimator-jsonl",
            str(estimator_path),
            "--reference-jsonl",
            str(reference_path),
            "--config-json",
            str(config_path),
            "--output-json",
            str(output_path),
        ]
    )

    assert return_code == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["valid"] is True
    assert result["gates_pass"] is True
    assert result["sample_counts"] == {
        "estimator": 9,
        "reference": 9,
        "waist_joint": 9,
    }
    assert len(result["inputs"]["estimator_sha256"]) == 64
