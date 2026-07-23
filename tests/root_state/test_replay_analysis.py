from __future__ import annotations

from dataclasses import replace

import pytest

from g1_root_state_bridge.protocol import (
    REQUIRED_HEALTH_FLAGS,
    RootStateHealth,
    RootStatePacketV2,
    serialize_root_state_v2,
)
from g1_root_state_bridge.replay_analysis import (
    ReplayGateConfig,
    analyze_packet_trace,
)


ALLOWED_DIGEST = bytes(range(32))


def _packet(**changes: object) -> RootStatePacketV2:
    packet = RootStatePacketV2(
        sequence=1,
        source_epoch=7,
        estimate_time_ns=10_000_000_000,
        publish_time_ns=10_001_000_000,
        correction_time_ns=9_900_000_000,
        joint_time_ns=9_999_000_000,
        joint_sync_gap_ns=-1_000_000,
        health_flags=REQUIRED_HEALTH_FLAGS,
        position=(0.0, 0.0, 0.8),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        covariance_diagonal=(1_000_000.0,) * 6,
        calibration_digest=ALLOWED_DIGEST,
    )
    return replace(packet, **changes)


def _record(packet: RootStatePacketV2, receipt_time_ns: int = 10_005_000_000) -> dict:
    return {
        "kind": "packet",
        "receipt_time_ns": receipt_time_ns,
        "payload_hex": serialize_root_state_v2(packet).hex(),
    }


def _config() -> ReplayGateConfig:
    return ReplayGateConfig(
        max_state_age_ns=10_000_000,
        max_correction_age_ns=150_000_000,
        allowed_calibration_digests=frozenset({ALLOWED_DIGEST}),
    )


def test_valid_trace_is_accepted_and_reports_timing() -> None:
    report = analyze_packet_trace(
        [_record(_packet())],
        config=_config(),
        trace_end_time_ns=10_006_000_000,
    )

    assert report.total_packets == 1
    assert report.accepted_packets == 1
    assert report.rejected_packets == 0
    assert report.rejection_counts == {}
    assert report.state_age_ns == [5_000_000]
    assert report.correction_age_ns == [105_000_000]
    assert report.publisher_loss is False


@pytest.mark.parametrize(
    ("packet", "receipt_time_ns", "reason"),
    [
        (
            _packet(
                estimate_time_ns=9_990_000_000,
                joint_time_ns=9_989_000_000,
            ),
            10_005_000_000,
            "state_stale",
        ),
        (
            _packet(correction_time_ns=9_800_000_000),
            10_005_000_000,
            "correction_stale",
        ),
        (
            _packet(
                health_flags=REQUIRED_HEALTH_FLAGS
                & ~RootStateHealth.JOINT_SYNC_VALID
            ),
            10_005_000_000,
            "health_flags_missing",
        ),
        (_packet(), 9_999_000_000, "clock_invalid"),
        (
            _packet(calibration_digest=b"x" * 32),
            10_005_000_000,
            "calibration_not_allowed",
        ),
    ],
)
def test_fault_packets_are_rejected(
    packet: RootStatePacketV2,
    receipt_time_ns: int,
    reason: str,
) -> None:
    report = analyze_packet_trace(
        [_record(packet, receipt_time_ns)],
        config=_config(),
        trace_end_time_ns=receipt_time_ns,
    )

    assert report.accepted_packets == 0
    assert report.rejection_counts == {reason: 1}


def test_malformed_packet_invalidates_the_previously_accepted_latest_state() -> None:
    records = [
        _record(_packet()),
        {
            "kind": "packet",
            "receipt_time_ns": 10_006_000_000,
            "payload_hex": b"malformed".hex(),
        },
    ]
    report = analyze_packet_trace(
        records,
        config=_config(),
        trace_end_time_ns=10_006_000_000,
    )

    assert report.accepted_packets == 1
    assert report.rejection_counts == {"malformed_payload": 1}
    assert report.latest_state_available is False


def test_epoch_reset_allows_sequence_restart_but_regression_is_rejected() -> None:
    records = [
        _record(_packet(source_epoch=7, sequence=8)),
        _record(_packet(source_epoch=8, sequence=1), 10_006_000_000),
        _record(_packet(source_epoch=7, sequence=100), 10_007_000_000),
    ]
    report = analyze_packet_trace(
        records,
        config=_config(),
        trace_end_time_ns=10_007_000_000,
    )

    assert report.accepted_packets == 2
    assert report.rejection_counts == {"source_epoch_not_increasing": 1}
    assert report.latest_state_available is False


def test_duplicate_sequence_is_rejected() -> None:
    records = [
        _record(_packet(sequence=4)),
        _record(_packet(sequence=4), 10_006_000_000),
    ]
    report = analyze_packet_trace(
        records,
        config=_config(),
        trace_end_time_ns=10_006_000_000,
    )

    assert report.rejection_counts == {"sequence_not_increasing": 1}


def test_publisher_loss_invalidates_latest_state_at_trace_end() -> None:
    report = analyze_packet_trace(
        [_record(_packet())],
        config=_config(),
        trace_end_time_ns=10_020_000_001,
    )

    assert report.publisher_loss is True
    assert report.latest_state_available is False
    assert report.rejection_counts == {"publisher_loss": 1}
