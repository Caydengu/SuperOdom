from __future__ import annotations

import numpy as np
import pytest
from g1_root_state_bridge.amo_dataset import (
    AffineClockMapping,
    LowStateCapture,
    detect_walking_boundary,
    fit_affine_lower_envelope_clock,
    load_motive,
)


def test_affine_clock_recovers_rate_and_lower_delay_envelope() -> None:
    source = (
        np.arange(0, 10_000_000_000, 10_000_000, dtype=np.int64) + 1_000_000_000_000
    )
    delay = np.tile(np.asarray([2_000_000, 2_500_000, 4_000_000, 9_000_000]), 250)
    receipt = np.rint(
        source[0] + 12_000_000 + 1.00001 * (source - source[0]) + delay
    ).astype(np.int64)
    mapping = fit_affine_lower_envelope_clock(source, receipt)
    # Repeated transport jitter can correlate weakly with source time; the
    # lower-envelope event mapping only needs clock-rate accuracy at a few ppm.
    assert abs(mapping.scale - 1.00001) < 5e-6
    event = mapping.map_ns(source)
    recovered_delay = receipt - event
    assert np.quantile(recovered_delay, 0.01) >= -1.0
    assert np.quantile(recovered_delay, 0.95) < 8_000_000


def _capture_with_leg_activity(onset_s: float) -> LowStateCapture:
    rate_hz = 100
    count = 20 * rate_hz
    event = (
        np.arange(count, dtype=np.int64) * (1_000_000_000 // rate_hz) + 10_000_000_000
    )
    velocity = np.zeros((count, 29), dtype=np.float32)
    velocity[event >= event[0] + round(onset_s * 1e9), :12] = 0.4
    zeros29 = np.zeros((count, 29), dtype=np.float32)
    clock = AffineClockMapping(0, 0, 1.0, 0.0, 0.0, 0.0, 0.0)
    return LowStateCapture(
        robot_stamp_ns=event,
        oslo_event_ns=event,
        receipt_realtime_ns=event + 1_000_000,
        sequence=np.arange(count, dtype=np.uint64),
        source_epoch=np.ones(count, dtype=np.uint64),
        source_tick=np.arange(count, dtype=np.uint32),
        joint_position=zeros29,
        joint_velocity=velocity,
        imu_quaternion_wxyz=np.tile(
            np.asarray([1, 0, 0, 0], dtype=np.float32), (count, 1)
        ),
        imu_gyroscope=np.zeros((count, 3), dtype=np.float32),
        imu_accelerometer=np.zeros((count, 3), dtype=np.float32),
        clock=clock,
    )


def test_walking_boundary_returns_start_of_first_sustained_window() -> None:
    boundary = detect_walking_boundary(_capture_with_leg_activity(7.3))
    assert boundary.seconds_from_lowstate_start == pytest.approx(7.3)
    assert boundary.event_realtime_ns == 17_300_000_000


def test_motive_identity_can_be_bound_from_a_run_manifest(tmp_path) -> None:
    source = tmp_path / "frames.jsonl"
    rows = [
        {
            "record_type": "metadata",
            "schema": "g1_optitrack_raw_v1",
            "requested_rigid_body_id": 42,
            "rigid_body_name": "G1_PELVIS_F_4123",
        }
    ]
    for index in range(3):
        rows.append(
            {
                "record_type": "frame",
                "rigid_body_id": 42,
                "rigid_body_name": "G1_PELVIS_F_4123",
                "tracking_valid": True,
                "mean_marker_error_m": 0.001,
                "motive_software_time_s": 10.0 + 0.01 * index,
                "receipt_realtime_ns": 20_000_000_000 + 10_000_000 * index,
                "frame_number": index,
                "position_xyz_m_motive_native": [0.0, 0.0, 0.0],
                "quaternion_xyzw_motive_native": [0.0, 0.0, 0.0, 1.0],
            }
        )
    source.write_text(
        "".join(__import__("json").dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    capture = load_motive(
        source,
        expected_rigid_body_id=42,
        expected_rigid_body_name="G1_PELVIS_F_4123",
    )
    assert capture.total_frames == 3
