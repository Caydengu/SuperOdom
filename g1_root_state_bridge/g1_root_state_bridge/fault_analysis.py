"""Score bounded live source faults against the root-state fail-closed contract."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _required_int(mapping: Mapping[str, object], key: str) -> int:
    try:
        value = int(mapping[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing or invalid {key}") from error
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    return value


def analyze_source_faults(
    records: Iterable[Mapping[str, object]],
    *,
    fault_events: Mapping[str, object],
    freshness_ns: Mapping[str, int],
) -> dict[str, Any]:
    """Return one deterministic report for joint and LiDAR source outages.

    A protected fault window begins strictly after the configured freshness
    deadline because a packet whose age equals the runtime limit remains valid.
    It ends when the restarted source reports ready.  No strict-valid packet may
    occur inside that interval.  Recovery is established independently by the
    first strict-valid packet at or after the ready timestamp.
    """
    packets: list[dict[str, object]] = []
    source_epochs: list[int] = []
    seen_epochs: set[int] = set()
    sequence_nonmonotonic_count = 0
    receipt_nonmonotonic_count = 0
    last_epoch: int | None = None
    last_sequence: int | None = None
    last_receipt: int | None = None

    for record in records:
        if record.get("event") != "packet_published":
            continue
        receipt = _required_int(record, "receipt_time_ns")
        sequence = _required_int(record, "sequence")
        epoch = _required_int(record, "source_epoch")
        strict = record.get("strictly_valid")
        if not isinstance(strict, bool):
            raise ValueError("packet strictly_valid must be a boolean")

        if last_receipt is not None and receipt <= last_receipt:
            receipt_nonmonotonic_count += 1
        if last_epoch is not None:
            if epoch < last_epoch or (
                epoch == last_epoch
                and last_sequence is not None
                and sequence <= last_sequence
            ):
                sequence_nonmonotonic_count += 1
        if epoch not in seen_epochs:
            seen_epochs.add(epoch)
            source_epochs.append(epoch)
        last_epoch = epoch
        last_sequence = sequence
        last_receipt = receipt
        packets.append(
            {
                "receipt_time_ns": receipt,
                "sequence": sequence,
                "source_epoch": epoch,
                "strictly_valid": strict,
                "joint_time_ns": _required_int(record, "joint_time_ns"),
                "correction_time_ns": _required_int(
                    record, "correction_time_ns"
                ),
            }
        )

    if not packets:
        raise ValueError("fault trace contains no packet_published records")

    fault_reports: dict[str, dict[str, int | bool | None]] = {}
    for source, limit_ns_raw in freshness_ns.items():
        try:
            event = fault_events[source]
        except KeyError as error:
            raise ValueError(f"fault event is missing {source}") from error
        if not isinstance(event, Mapping):
            raise ValueError(f"fault event {source} must be a mapping")
        limit_ns = int(limit_ns_raw)
        if limit_ns <= 0:
            raise ValueError(f"freshness limit for {source} must be positive")
        stopped_ns = _required_int(event, "stopped_ns")
        restart_request_ns = _required_int(event, "restart_request_ns")
        restarted_ready_ns = _required_int(event, "restarted_ready_ns")
        if not stopped_ns < restart_request_ns <= restarted_ready_ns:
            raise ValueError(f"{source} restart must occur after stop")
        freshness_deadline_ns = stopped_ns + limit_ns
        if freshness_deadline_ns >= restart_request_ns:
            raise ValueError(
                f"{source} outage does not extend beyond its freshness limit"
            )

        evidence_field = {
            "joint_source": "joint_time_ns",
            "lidar_source": "correction_time_ns",
        }.get(source)
        if evidence_field is None:
            raise ValueError(f"unsupported source fault {source}")

        protected = [
            packet
            for packet in packets
            if freshness_deadline_ns < int(packet["receipt_time_ns"])
            < restart_request_ns
        ]
        strict_violations = sum(
            bool(packet["strictly_valid"]) for packet in protected
        )
        old_evidence_strict = sum(
            bool(packet["strictly_valid"])
            and int(packet["receipt_time_ns"]) >= restart_request_ns
            and int(packet[evidence_field]) < restart_request_ns
            for packet in packets
        )
        first_valid_after = next(
            (
                packet
                for packet in packets
                if int(packet["receipt_time_ns"]) >= restart_request_ns
                and bool(packet["strictly_valid"])
                and int(packet[evidence_field]) >= restart_request_ns
            ),
            None,
        )
        first_valid_receipt_ns = (
            None
            if first_valid_after is None
            else int(first_valid_after["receipt_time_ns"])
        )
        first_fresh_evidence_ns = (
            None
            if first_valid_after is None
            else int(first_valid_after[evidence_field])
        )
        fail_closed = strict_violations == 0 and old_evidence_strict == 0
        recovered = first_valid_after is not None
        fault_reports[source] = {
            "stopped_ns": stopped_ns,
            "freshness_deadline_ns": freshness_deadline_ns,
            "restart_request_ns": restart_request_ns,
            "restarted_ready_ns": restarted_ready_ns,
            "strict_valid_in_protected_window": strict_violations,
            "packets_in_protected_window": len(protected),
            "invalid_packets_in_protected_window": sum(
                not bool(packet["strictly_valid"]) for packet in protected
            ),
            "strict_valid_with_old_evidence_after_restart": old_evidence_strict,
            "first_strict_valid_after_fresh_evidence_ns": first_valid_receipt_ns,
            "first_fresh_evidence_time_ns": first_fresh_evidence_ns,
            "recovery_latency_from_request_ns": (
                None
                if first_valid_receipt_ns is None
                else first_valid_receipt_ns - restart_request_ns
            ),
            "recovered_before_ready_check": (
                False
                if first_valid_receipt_ns is None
                else first_valid_receipt_ns < restarted_ready_ns
            ),
            "fail_closed": fail_closed,
            "recovered": recovered,
            "window_pass": fail_closed and recovered,
        }

    all_fault_windows_pass = (
        len(source_epochs) == 1
        and sequence_nonmonotonic_count == 0
        and receipt_nonmonotonic_count == 0
        and all(bool(report["window_pass"]) for report in fault_reports.values())
    )
    return {
        "packet_count": len(packets),
        "source_epochs": source_epochs,
        "sequence_nonmonotonic_count": sequence_nonmonotonic_count,
        "receipt_nonmonotonic_count": receipt_nonmonotonic_count,
        "faults": fault_reports,
        "all_fault_windows_pass": all_fault_windows_pass,
    }
