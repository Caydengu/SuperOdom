from __future__ import annotations

import math

from g1_root_state_bridge.optitrack_clock_audit import (
    audit_optitrack_clock_records,
)


def _clock_records(*, jitter_ns: int = 200_000) -> list[dict[str, object]]:
    records: list[dict[str, object]] = [
        {
            "schema": "g1_optitrack_raw_v1",
            "kind": "metadata",
            "high_resolution_clock_frequency_hz": 10_000_000,
        }
    ]
    source_start_s = 100.0
    target_offset_ns = 1_700_000_000_000_000_000
    for index in range(240):
        source_s = source_start_s + index / 120.0
        jitter = int(jitter_ns * math.sin(index * 0.31))
        capture_ns = int(source_s * 1e9) + target_offset_ns + jitter
        mid_ticks = 50_000_000 + index * 83_333
        records.append(
            {
                "schema": "g1_optitrack_raw_v1",
                "kind": "raw_rigid_body_pose",
                "frame_number": 1000 + index,
                "motive_software_time_s": source_s,
                "camera_mid_exposure_ticks": mid_ticks,
                "camera_data_received_ticks": mid_ticks + 20_000,
                "transmit_ticks": mid_ticks + 30_000,
                "high_resolution_clock_frequency_hz": 10_000_000,
                "capture_realtime_estimate_ns": capture_ns,
                "receipt_realtime_ns": capture_ns + 4_000_000,
                "seconds_since_host_mid_exposure": 0.004,
            }
        )
    return records


def test_stable_clock_mapping_passes_and_returns_affine_contract() -> None:
    report = audit_optitrack_clock_records(
        _clock_records(),
        maximum_residual_p95_ns=5_000_000,
        minimum_frames=100,
        calibration_id="natnet-clock-audit-v1",
    )

    assert report.valid is True
    assert report.gates_pass is True
    assert report.invalid_reasons == ()
    assert report.metrics["usable_frame_count"] == 240
    assert report.metrics["mapping_residual_p95_ns"] < 250_000
    assert abs(report.metrics["mapping_scale_drift_ppm"]) < 100.0
    assert report.metrics["client_latency_p95_ms"] == 4.0
    assert report.metrics["system_latency_p95_ms"] == 3.0
    assert report.clock_mapping is not None
    assert report.clock_mapping.source_clock_id == "motive_software_ns"
    assert report.clock_mapping.target_clock_id == "unix_realtime_ns"
    assert report.clock_mapping.calibration_id == "natnet-clock-audit-v1"


def test_excessive_mapping_jitter_fails_clock_gate() -> None:
    report = audit_optitrack_clock_records(
        _clock_records(jitter_ns=7_000_000),
        maximum_residual_p95_ns=5_000_000,
        minimum_frames=100,
        calibration_id="bad-clock-audit",
    )

    assert report.valid is True
    assert report.gates_pass is False
    assert report.gate_results["mapping_residual_p95"] is False
    assert report.metrics["mapping_residual_p95_ns"] > 5_000_000


def test_nonmonotonic_frame_or_timestamp_invalidates_evidence() -> None:
    records = _clock_records()
    records[80]["frame_number"] = records[79]["frame_number"]
    records[120]["capture_realtime_estimate_ns"] = (
        records[119]["capture_realtime_estimate_ns"] - 1
    )

    report = audit_optitrack_clock_records(
        records,
        maximum_residual_p95_ns=5_000_000,
        minimum_frames=100,
        calibration_id="invalid-clock-audit",
    )

    assert report.valid is False
    assert "frame_number_not_strictly_increasing" in report.invalid_reasons
    assert "capture_time_not_strictly_increasing" in report.invalid_reasons
    assert report.gates_pass is False


def test_missing_capture_timestamps_cannot_satisfy_minimum_frames() -> None:
    records = _clock_records()
    for record in records[2:]:
        record["capture_realtime_estimate_ns"] = None

    report = audit_optitrack_clock_records(
        records,
        maximum_residual_p95_ns=5_000_000,
        minimum_frames=100,
        calibration_id="missing-clock-audit",
    )

    assert report.valid is False
    assert "insufficient_clock_frames" in report.invalid_reasons

