"""Deterministic strict-gate analysis for recorded root-state V2 payloads.

The analyzer mirrors the deployed Holosoma acceptance contract while retaining
aggregate timing evidence.  Any rejected payload clears the modeled latest
state, which is the fail-closed behavior required for perceptive-real use.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from g1_root_state_bridge.protocol import (
    REQUIRED_HEALTH_FLAGS,
    RootStatePacketV2,
    RootStateProtocolError,
    deserialize_root_state_v2,
)


@dataclass(frozen=True)
class ReplayGateConfig:
    max_state_age_ns: int
    max_correction_age_ns: int
    allowed_calibration_digests: frozenset[bytes]

    def __post_init__(self) -> None:
        if self.max_state_age_ns <= 0 or self.max_correction_age_ns <= 0:
            raise ValueError("replay age thresholds must be positive")
        if not self.allowed_calibration_digests:
            raise ValueError("at least one calibration digest must be allowed")
        if any(len(digest) != 32 for digest in self.allowed_calibration_digests):
            raise ValueError("calibration digests must contain 32 bytes")


@dataclass
class ReplayTraceReport:
    total_packets: int = 0
    accepted_packets: int = 0
    rejected_packets: int = 0
    rejection_counts: dict[str, int] = field(default_factory=dict)
    state_age_ns: list[int] = field(default_factory=list)
    correction_age_ns: list[int] = field(default_factory=list)
    joint_sync_gap_ns: list[int] = field(default_factory=list)
    publisher_loss: bool = False
    latest_state_available: bool = False
    last_source_epoch: int | None = None
    last_sequence: int | None = None


def _rejection_reason(
    packet: RootStatePacketV2,
    *,
    receipt_time_ns: int,
    config: ReplayGateConfig,
    last_source_epoch: int | None,
    last_sequence: int | None,
) -> str | None:
    if (packet.health_flags & REQUIRED_HEALTH_FLAGS) != REQUIRED_HEALTH_FLAGS:
        return "health_flags_missing"
    if packet.calibration_digest not in config.allowed_calibration_digests:
        return "calibration_not_allowed"

    state_age_ns = receipt_time_ns - packet.estimate_time_ns
    correction_age_ns = receipt_time_ns - packet.correction_time_ns
    if (
        state_age_ns < 0
        or correction_age_ns < 0
        or receipt_time_ns < packet.publish_time_ns
    ):
        return "clock_invalid"
    if state_age_ns > config.max_state_age_ns:
        return "state_stale"
    if correction_age_ns > config.max_correction_age_ns:
        return "correction_stale"

    if last_source_epoch is not None:
        if packet.source_epoch < last_source_epoch:
            return "source_epoch_not_increasing"
        if packet.source_epoch == last_source_epoch:
            assert last_sequence is not None
            if packet.sequence <= last_sequence:
                return "sequence_not_increasing"
    return None


def analyze_packet_trace(
    records: Iterable[Mapping[str, object]],
    *,
    config: ReplayGateConfig,
    trace_end_time_ns: int,
) -> ReplayTraceReport:
    """Analyze packet JSONL records under the exact strict runtime semantics."""
    if trace_end_time_ns < 0:
        raise ValueError("trace_end_time_ns must be non-negative")

    report = ReplayTraceReport()
    rejection_counts: Counter[str] = Counter()
    latest_packet: RootStatePacketV2 | None = None

    for record in records:
        if record.get("kind") != "packet":
            continue
        report.total_packets += 1
        try:
            receipt_time_ns = int(record["receipt_time_ns"])
            payload_hex = record["payload_hex"]
            if not isinstance(payload_hex, str):
                raise TypeError("payload_hex is not a string")
            packet = deserialize_root_state_v2(bytes.fromhex(payload_hex))
        except (KeyError, TypeError, ValueError, RootStateProtocolError):
            report.rejected_packets += 1
            rejection_counts["malformed_payload"] += 1
            latest_packet = None
            continue

        reason = _rejection_reason(
            packet,
            receipt_time_ns=receipt_time_ns,
            config=config,
            last_source_epoch=report.last_source_epoch,
            last_sequence=report.last_sequence,
        )
        if reason is not None:
            report.rejected_packets += 1
            rejection_counts[reason] += 1
            latest_packet = None
            continue

        report.accepted_packets += 1
        report.state_age_ns.append(receipt_time_ns - packet.estimate_time_ns)
        report.correction_age_ns.append(receipt_time_ns - packet.correction_time_ns)
        report.joint_sync_gap_ns.append(packet.joint_sync_gap_ns)
        report.last_source_epoch = packet.source_epoch
        report.last_sequence = packet.sequence
        latest_packet = packet

    if latest_packet is not None:
        state_age_at_end_ns = trace_end_time_ns - latest_packet.estimate_time_ns
        correction_age_at_end_ns = trace_end_time_ns - latest_packet.correction_time_ns
        if (
            state_age_at_end_ns < 0
            or correction_age_at_end_ns < 0
            or trace_end_time_ns < latest_packet.publish_time_ns
        ):
            rejection_counts["clock_invalid"] += 1
            latest_packet = None
        elif state_age_at_end_ns > config.max_state_age_ns:
            rejection_counts["publisher_loss"] += 1
            report.publisher_loss = True
            latest_packet = None
        elif correction_age_at_end_ns > config.max_correction_age_ns:
            rejection_counts["correction_stale"] += 1
            latest_packet = None

    report.rejection_counts = dict(sorted(rejection_counts.items()))
    report.latest_state_available = latest_packet is not None
    return report
