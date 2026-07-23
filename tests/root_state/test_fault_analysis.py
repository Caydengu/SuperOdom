from __future__ import annotations

from g1_root_state_bridge.fault_analysis import analyze_source_faults


def _packet(
    receipt_time_ns: int,
    sequence: int,
    *,
    strict: bool,
    source_epoch: int = 7,
    joint_time_ns: int | None = None,
    correction_time_ns: int | None = None,
) -> dict[str, object]:
    return {
        "event": "packet_published",
        "kind": "packet",
        "receipt_time_ns": receipt_time_ns,
        "sequence": sequence,
        "source_epoch": source_epoch,
        "strictly_valid": strict,
        "joint_time_ns": (
            receipt_time_ns if joint_time_ns is None else joint_time_ns
        ),
        "correction_time_ns": (
            receipt_time_ns if correction_time_ns is None else correction_time_ns
        ),
    }


def _fault_events() -> dict[str, object]:
    return {
        "joint_source": {
            "stopped_ns": 1_000,
            "restart_request_ns": 1_900,
            "restarted_ready_ns": 2_000,
        },
        "lidar_source": {
            "stopped_ns": 3_000,
            "restart_request_ns": 3_900,
            "restarted_ready_ns": 4_000,
        },
    }


def test_each_source_fails_closed_after_its_deadline_and_recovers() -> None:
    records = [
        _packet(900, 1, strict=True),
        _packet(1_005, 2, strict=True),
        _packet(1_011, 3, strict=False),
        _packet(1_500, 4, strict=False),
        _packet(1_950, 5, strict=True, joint_time_ns=1_940),
        _packet(2_900, 6, strict=True),
        _packet(3_200, 7, strict=True),
        _packet(3_251, 8, strict=False),
        _packet(3_900, 9, strict=False),
        _packet(4_300, 10, strict=True),
    ]

    report = analyze_source_faults(
        records,
        fault_events=_fault_events(),
        freshness_ns={"joint_source": 10, "lidar_source": 250},
    )

    assert report["all_fault_windows_pass"] is True
    assert report["sequence_nonmonotonic_count"] == 0
    assert report["source_epochs"] == [7]
    assert report["faults"]["joint_source"] == {
        "stopped_ns": 1_000,
        "freshness_deadline_ns": 1_010,
        "restart_request_ns": 1_900,
        "restarted_ready_ns": 2_000,
        "strict_valid_in_protected_window": 0,
        "packets_in_protected_window": 2,
        "invalid_packets_in_protected_window": 2,
        "strict_valid_with_old_evidence_after_restart": 0,
        "first_strict_valid_after_fresh_evidence_ns": 1_950,
        "first_fresh_evidence_time_ns": 1_940,
        "recovery_latency_from_request_ns": 50,
        "recovered_before_ready_check": True,
        "fail_closed": True,
        "recovered": True,
        "window_pass": True,
    }
    assert report["faults"]["lidar_source"][
        "first_strict_valid_after_fresh_evidence_ns"
    ] == 4_300
    assert report["faults"]["lidar_source"][
        "recovery_latency_from_request_ns"
    ] == 400


def test_strict_packet_after_freshness_deadline_is_a_safety_violation() -> None:
    records = [
        _packet(900, 1, strict=True),
        _packet(1_050, 2, strict=True),
        _packet(2_100, 3, strict=True),
        _packet(2_900, 4, strict=True),
        _packet(3_251, 5, strict=False),
        _packet(4_100, 6, strict=True),
    ]

    report = analyze_source_faults(
        records,
        fault_events=_fault_events(),
        freshness_ns={"joint_source": 10, "lidar_source": 250},
    )

    assert report["all_fault_windows_pass"] is False
    assert report["faults"]["joint_source"][
        "strict_valid_in_protected_window"
    ] == 1
    assert report["faults"]["joint_source"]["fail_closed"] is False


def test_missing_post_restart_packet_does_not_claim_recovery() -> None:
    records = [
        _packet(900, 1, strict=True),
        _packet(1_500, 2, strict=False),
        _packet(2_900, 3, strict=True),
        _packet(3_500, 4, strict=False),
    ]

    report = analyze_source_faults(
        records,
        fault_events=_fault_events(),
        freshness_ns={"joint_source": 10, "lidar_source": 250},
    )

    assert report["faults"]["lidar_source"]["recovered"] is False
    assert report["faults"]["lidar_source"]["window_pass"] is False
    assert report["all_fault_windows_pass"] is False


def test_strict_packet_with_pre_restart_evidence_does_not_count_as_recovery() -> None:
    records = [
        _packet(900, 1, strict=True),
        _packet(1_500, 2, strict=False),
        _packet(1_950, 3, strict=True, joint_time_ns=900),
        _packet(2_900, 4, strict=True),
        _packet(3_500, 5, strict=False),
        _packet(4_100, 6, strict=True),
    ]

    report = analyze_source_faults(
        records,
        fault_events=_fault_events(),
        freshness_ns={"joint_source": 10, "lidar_source": 250},
    )

    joint = report["faults"]["joint_source"]
    assert joint["strict_valid_with_old_evidence_after_restart"] == 1
    assert joint["recovered"] is True
    assert joint["first_strict_valid_after_fresh_evidence_ns"] == 2_900
    assert joint["window_pass"] is False


def test_nonmonotonic_packet_sequence_fails_the_trial() -> None:
    records = [
        _packet(900, 2, strict=True),
        _packet(1_500, 1, strict=False),
        _packet(2_100, 3, strict=True),
        _packet(2_900, 4, strict=True),
        _packet(3_500, 5, strict=False),
        _packet(4_100, 6, strict=True),
    ]

    report = analyze_source_faults(
        records,
        fault_events=_fault_events(),
        freshness_ns={"joint_source": 10, "lidar_source": 250},
    )

    assert report["sequence_nonmonotonic_count"] == 1
    assert report["all_fault_windows_pass"] is False
