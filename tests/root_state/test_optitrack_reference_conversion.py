from __future__ import annotations

import json

import pytest

from g1_root_state_bridge.dynamic_reference_io import load_reference_pose_jsonl
from g1_root_state_bridge.optitrack_reference import (
    OptiTrackReferenceError,
    convert_raw_optitrack_records,
    load_pelvis_calibration,
)


def _calibration_payload(**changes: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "g1_optitrack_pelvis_calibration_v1",
        "status": "validated",
        "calibration_id": "direct-pelvis-rigid-body-v1",
        "raw_capture_calibration_id": "motive-rigid-body-definition-v7",
        "rigid_body_id": 10,
        "rigid_body_name": "g1_pelvis",
        "tracked_link": "pelvis",
        "target_frame_id": "mocap_world_z_up",
        "clock_id": "unix_realtime_ns",
        "clock_mapping_method": "natnet_seconds_since_host_timestamp",
        "clock_mapping_calibration_id": "natnet-cristian-audit-v1",
        "clock_mapping_residual_p95_ns": 1_000_000,
        "max_clock_mapping_residual_ns": 5_000_000,
        "maximum_marker_error_m": 0.005,
        "maximum_velocity_interval_ns": 150_000_000,
        "T_target_motive_world": {
            "translation_m": [0.0, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "T_rigid_body_pelvis": {
            "translation_m": [0.0, 0.0, 0.2],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    }
    payload.update(changes)
    return payload


def _raw_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = [
        {
            "schema": "g1_optitrack_raw_v1",
            "kind": "metadata",
            "pose_semantics": "rigid_body_pose_raw",
            "coordinate_frame": "motive_world_native",
            "rigid_body_id": 10,
            "rigid_body_name": "g1_pelvis",
            "rigid_body_identity_validated": True,
            "server_rigid_body_name": "g1_pelvis",
            "server_rigid_body_marker_count": 3,
            "server_rigid_body_markers": [
                {
                    "index": 0,
                    "name": "pelvis_a",
                    "position_xyz_m": [0.0, 0.0, 0.0],
                    "required_active_label": 0,
                },
                {
                    "index": 1,
                    "name": "pelvis_b",
                    "position_xyz_m": [0.1, 0.0, 0.0],
                    "required_active_label": 0,
                },
                {
                    "index": 2,
                    "name": "pelvis_c",
                    "position_xyz_m": [0.0, 0.1, 0.0],
                    "required_active_label": 0,
                },
            ],
            "calibration_id": "motive-rigid-body-definition-v7",
            "natnet_sdk_archive_sha256": "a" * 64,
        }
    ]
    for index in range(5):
        records.append(
            {
                "schema": "g1_optitrack_raw_v1",
                "kind": "raw_rigid_body_pose",
                "frame_number": 100 + index,
                "motive_software_time_s": 12.0 + 0.1 * index,
                "camera_mid_exposure_ticks": 1000 + index,
                "camera_data_received_ticks": 1100 + index,
                "transmit_ticks": 1200 + index,
                "precision_timestamp_seconds": 0,
                "precision_timestamp_fractional_seconds": 0,
                "receipt_realtime_ns": 1_005_000_000 + index * 100_000_000,
                "receipt_monotonic_ns": 5_000_000 + index * 100_000_000,
                "high_resolution_clock_frequency_hz": 10_000_000,
                "seconds_since_host_mid_exposure": 0.005,
                "capture_realtime_estimate_ns": (
                    1_000_000_000 + index * 100_000_000
                ),
                "rigid_body_id": 10,
                "rigid_body_present": True,
                "tracking_valid": True,
                "mean_marker_error_m": 0.0005,
                "position_motive_xyz_m": [0.1 * index, 0.0, 0.0],
                "quaternion_motive_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        )
    return records


def test_direct_pelvis_conversion_applies_explicit_transforms_and_velocity(
    tmp_path,
) -> None:
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        json.dumps(_calibration_payload()),
        encoding="utf-8",
    )
    calibration = load_pelvis_calibration(calibration_path)

    normalized = convert_raw_optitrack_records(
        _raw_records(),
        calibration=calibration,
        calibration_sha256="b" * 64,
    )

    assert normalized[0]["schema"] == "g1_dynamic_reference_v1"
    assert normalized[0]["pose_semantics"] == "pelvis_pose"
    assert normalized[0]["clock_id"] == "unix_realtime_ns"
    assert normalized[0]["calibration_id"] == "direct-pelvis-rigid-body-v1"
    middle = normalized[3]
    assert middle["position_m"] == pytest.approx([0.2, 0.0, 0.2])
    assert middle["quaternion_wxyz"] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert middle["linear_velocity_mps"] == pytest.approx([1.0, 0.0, 0.0])
    assert middle["angular_velocity_radps"] == pytest.approx([0.0, 0.0, 0.0])

    output = tmp_path / "normalized.jsonl"
    output.write_text(
        "".join(json.dumps(record) + "\n" for record in normalized),
        encoding="utf-8",
    )
    loaded = load_reference_pose_jsonl(output)
    assert len(loaded) == 5
    assert loaded[2].position == pytest.approx((0.2, 0.0, 0.2))
    assert loaded[2].linear_velocity == pytest.approx((1.0, 0.0, 0.0))


def test_tracking_loss_is_preserved_as_invalid_not_interpolated_truth() -> None:
    records = _raw_records()
    records[3]["tracking_valid"] = False
    records[3]["position_motive_xyz_m"] = None
    records[3]["quaternion_motive_xyzw"] = None
    calibration = load_pelvis_calibration(_calibration_payload())

    normalized = convert_raw_optitrack_records(
        records,
        calibration=calibration,
        calibration_sha256="b" * 64,
    )

    lost = normalized[3]
    assert lost["tracking_valid"] is False
    assert lost["position_m"] is None
    assert lost["quaternion_wxyz"] is None
    assert "linear_velocity_mps" not in lost


def test_marker_error_above_frozen_limit_invalidates_only_that_sample() -> None:
    records = _raw_records()
    records[2]["mean_marker_error_m"] = 0.006
    calibration = load_pelvis_calibration(_calibration_payload())

    normalized = convert_raw_optitrack_records(
        records,
        calibration=calibration,
        calibration_sha256="b" * 64,
    )

    assert normalized[2]["tracking_valid"] is False
    assert normalized[2]["invalid_reason"] == "marker_error_exceeds_limit"


def test_converter_refuses_head_link_and_unvalidated_clock() -> None:
    head_payload = _calibration_payload(tracked_link="head_link")
    with pytest.raises(OptiTrackReferenceError, match="direct pelvis"):
        load_pelvis_calibration(head_payload)

    bad_clock = _calibration_payload(clock_mapping_residual_p95_ns=6_000_000)
    with pytest.raises(OptiTrackReferenceError, match="clock"):
        load_pelvis_calibration(bad_clock)


def test_raw_capture_identity_must_match_calibration() -> None:
    records = _raw_records()
    records[0]["rigid_body_id"] = 99
    calibration = load_pelvis_calibration(_calibration_payload())

    with pytest.raises(OptiTrackReferenceError, match="identity"):
        convert_raw_optitrack_records(
            records,
            calibration=calibration,
            calibration_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rigid_body_identity_validated", False),
        ("server_rigid_body_name", "not_the_pelvis"),
        ("server_rigid_body_marker_count", 2),
    ],
)
def test_raw_capture_requires_server_validated_body_definition(
    field: str,
    value: object,
) -> None:
    records = _raw_records()
    records[0][field] = value
    calibration = load_pelvis_calibration(_calibration_payload())

    with pytest.raises(OptiTrackReferenceError, match="Motive"):
        convert_raw_optitrack_records(
            records,
            calibration=calibration,
            calibration_sha256="b" * 64,
        )
