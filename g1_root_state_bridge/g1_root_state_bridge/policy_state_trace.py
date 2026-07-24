"""Convert retained bridge evidence into strict policy-state replay packets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from g1_root_state_bridge.joint_contract import (
    CANONICAL_G1_JOINT_NAMES,
    JointContractError,
    TimedJointSample,
)
from g1_root_state_bridge.policy_state_protocol import (
    G1PolicyStatePacketV1,
    PolicyStateHealth,
    PolicyStateProtocolError,
    build_policy_state_v1,
    deserialize_policy_state_v1,
    serialize_policy_state_v1,
)
from g1_root_state_bridge.protocol import (
    RootStateProtocolError,
    deserialize_root_state_v2,
)


class PolicyStateTraceError(ValueError):
    """Retained evidence cannot produce a strict policy-state replay."""

    def __init__(
        self,
        message: str,
        *,
        report: ReplayConversionReport | None = None,
    ) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class RejectedCandidateInterval:
    """One maximal attempted replay interval rejected by a frozen gate."""

    query_start_ns: int
    query_end_ns: int
    samples_before_rejection: int
    reason: str


@dataclass(frozen=True)
class ReplayConversionReport:
    """Bounded evidence and identity receipt for one replay conversion."""

    samples: int
    source_records_read: int
    duplicate_sequences: int
    root_age_p95_ms: float
    correction_age_p95_ms: float
    joint_sync_abs_p95_ms: float
    joint_sync_abs_max_ms: float
    calibration_digest_sha256: str
    joint_mapping_digest_sha256: str
    selected_query_start_ns: int | None
    selected_query_end_ns: int | None
    selected_source_publish_start_ns: int | None
    selected_source_publish_end_ns: int | None
    rejected_candidate_intervals: tuple[RejectedCandidateInterval, ...]

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def _direct_state(packet: G1PolicyStatePacketV1) -> dict[str, object]:
    return {
        "sequence": packet.sequence,
        "source_epoch": packet.source_epoch,
        "estimate_time_ns": packet.estimate_time_ns,
        "publish_time_ns": packet.publish_time_ns,
        "correction_time_ns": packet.correction_time_ns,
        "joint_time_ns": packet.joint_time_ns,
        "joint_sync_gap_ns": packet.joint_sync_gap_ns,
        "health_flags": int(packet.health_flags),
        "position": list(packet.position),
        "quaternion_wxyz": list(packet.quaternion_wxyz),
        "linear_velocity": list(packet.linear_velocity),
        "angular_velocity": list(packet.angular_velocity),
        "covariance_diagonal": list(packet.covariance_diagonal),
        "joint_position": list(packet.joint_position),
        "joint_velocity": list(packet.joint_velocity),
    }


def write_simulation_pair(
    output_path: Path,
    *,
    calibration_digest: bytes,
    joint_mapping_digest: bytes,
    samples: int = 2,
) -> None:
    """Write deterministic direct-plus-wire states for parity/load tests."""

    if len(calibration_digest) != 32 or len(joint_mapping_digest) != 32:
        raise PolicyStateTraceError(
            "simulation-pair digests must contain 32 bytes"
        )
    if not isinstance(samples, int) or samples < 2:
        raise PolicyStateTraceError(
            "simulation fixture requires at least two samples"
        )
    base_ns = 1_700_000_000_000_000_000
    packets = []
    for tick in range(samples):
        sequence = 101 + tick
        query_time_ns = base_ns + tick * 20_000_000
        estimate_time_ns = query_time_ns - 2_000_000
        packet = G1PolicyStatePacketV1(
            sequence=sequence,
            source_epoch=7,
            estimate_time_ns=estimate_time_ns,
            publish_time_ns=query_time_ns - 1_000_000,
            correction_time_ns=query_time_ns - 100_000_000,
            joint_time_ns=estimate_time_ns + 250_000,
            joint_sync_gap_ns=250_000,
            health_flags=PolicyStateHealth(0xFF),
            position=(
                0.0001 * tick,
                -0.0001 * tick,
                0.73 + 0.0001 * (tick % 10),
            ),
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            linear_velocity=(
                0.001 * (tick % 10),
                -0.001 * (tick % 10),
                0.0,
            ),
            angular_velocity=(
                0.001,
                0.002,
                0.003 + 0.0001 * (tick % 10),
            ),
            covariance_diagonal=(0.01,) * 6,
            joint_position=tuple(
                (index - 14) / 100.0 + tick / 10000.0
                for index in range(29)
            ),
            joint_velocity=tuple(
                (14 - index) / 1000.0 + (tick % 10) / 10000.0
                for index in range(29)
            ),
            calibration_digest=calibration_digest,
            joint_mapping_digest=joint_mapping_digest,
        )
        payload = serialize_policy_state_v1(packet)
        packets.append((tick, query_time_ns, packet, payload))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        for tick, query_time_ns, packet, payload in packets:
            output.write(
                json.dumps(
                    {
                        "schema": "g1_policy_state_replay_v1",
                        "tick": tick,
                        "query_time_ns": query_time_ns,
                        "sequence": packet.sequence,
                        "source_epoch": packet.source_epoch,
                        "estimate_time_ns": packet.estimate_time_ns,
                        "publish_time_ns": packet.publish_time_ns,
                        "correction_time_ns": packet.correction_time_ns,
                        "joint_time_ns": packet.joint_time_ns,
                        "payload_hex": payload.hex(),
                        "direct_state": _direct_state(packet),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _finite_29(value: object, name: str) -> tuple[float, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 29
    ):
        raise PolicyStateTraceError(f"{name} must contain 29 finite values")
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise PolicyStateTraceError(
            f"{name} must contain 29 finite values"
        ) from error
    if not all(math.isfinite(item) for item in values):
        raise PolicyStateTraceError(f"{name} must contain 29 finite values")
    return values


def policy_state_from_status_record(
    record: Mapping[str, object],
    *,
    expected_calibration_digest: bytes,
    expected_joint_mapping_digest: bytes,
) -> G1PolicyStatePacketV1:
    """Reconstruct policy state from serialized root and synchronized q/dq."""
    payload_hex = record.get("payload_hex")
    if not isinstance(payload_hex, str):
        raise PolicyStateTraceError("packet record lacks root payload_hex")
    try:
        root = deserialize_root_state_v2(bytes.fromhex(payload_hex))
    except (ValueError, RootStateProtocolError) as error:
        raise PolicyStateTraceError("packet record has malformed root payload") from error

    recorded_strict = record.get("strictly_valid")
    if not isinstance(recorded_strict, bool):
        raise PolicyStateTraceError("packet record lacks boolean strict validity")
    if recorded_strict != root.strictly_valid:
        raise PolicyStateTraceError(
            "recorded strict validity differs from the root payload"
        )
    if root.calibration_digest != expected_calibration_digest:
        raise PolicyStateTraceError("root calibration digest mismatch")
    if record.get("joint_mapping_digest_sha256") != (
        expected_joint_mapping_digest.hex()
    ):
        raise PolicyStateTraceError("joint mapping digest mismatch")
    if record.get("joint_names") != list(CANONICAL_G1_JOINT_NAMES):
        raise PolicyStateTraceError("joint names are not canonical")

    position = _finite_29(record.get("joint_position"), "joint_position")
    velocity = _finite_29(record.get("joint_velocity"), "joint_velocity")
    try:
        sample = TimedJointSample(
            stamp_ns=root.estimate_time_ns,
            receipt_ns=root.publish_time_ns,
            names=CANONICAL_G1_JOINT_NAMES,
            position=position,
            velocity=velocity,
            sequence=root.sequence,
            source_epoch=root.source_epoch,
        )
        synthesized = build_policy_state_v1(
            root,
            sample,
            joint_mapping_digest=expected_joint_mapping_digest,
        )
        synthesized_payload = serialize_policy_state_v1(synthesized)
    except (JointContractError, PolicyStateProtocolError) as error:
        raise PolicyStateTraceError(
            "root-plus-joint evidence violates the policy-state contract"
        ) from error

    recorded_policy_hex = record.get("policy_state_payload_hex")
    if recorded_policy_hex is None:
        return synthesized
    if not isinstance(recorded_policy_hex, str):
        raise PolicyStateTraceError(
            "policy_state_payload_hex must be a string"
        )
    try:
        recorded_payload = bytes.fromhex(recorded_policy_hex)
        recorded_policy = deserialize_policy_state_v1(recorded_payload)
    except (ValueError, PolicyStateProtocolError) as error:
        raise PolicyStateTraceError(
            "recorded policy packet is malformed"
        ) from error
    if recorded_payload != synthesized_payload:
        raise PolicyStateTraceError(
            "recorded policy packet differs from root-plus-joint evidence"
        )
    return recorded_policy


@dataclass
class _SourceStats:
    records_read: int = 0


def _iter_policy_packets(
    path: Path,
    *,
    expected_calibration_digest: bytes,
    expected_joint_mapping_digest: bytes,
    stats: _SourceStats,
) -> Iterator[G1PolicyStatePacketV1]:
    previous_publish_time_ns: int | None = None
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            stats.records_read += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise PolicyStateTraceError(
                    f"invalid JSON on source line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise PolicyStateTraceError(
                    f"source line {line_number} is not a JSON object"
                )
            if (
                record.get("event") != "packet_published"
                or record.get("kind") != "packet"
            ):
                continue
            packet = policy_state_from_status_record(
                record,
                expected_calibration_digest=expected_calibration_digest,
                expected_joint_mapping_digest=expected_joint_mapping_digest,
            )
            if (
                previous_publish_time_ns is not None
                and packet.publish_time_ns < previous_publish_time_ns
            ):
                raise PolicyStateTraceError(
                    "source packet publish times are not monotonic"
                )
            previous_publish_time_ns = packet.publish_time_ns
            yield packet


def _percentile_95(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = 0.95 * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _candidate_rejection(
    packet: G1PolicyStatePacketV1,
    *,
    query_time_ns: int,
    previous_packet: G1PolicyStatePacketV1 | None,
) -> str | None:
    if not packet.strictly_valid:
        return "policy health flags are incomplete"
    if packet.publish_time_ns > query_time_ns:
        return "packet publication is noncausal"
    root_age_ns = query_time_ns - packet.estimate_time_ns
    if not 0 <= root_age_ns <= 10_000_000:
        return "replayed root state is outside the 10 ms gate"
    correction_age_ns = query_time_ns - packet.correction_time_ns
    if not 0 <= correction_age_ns <= 250_000_000:
        return "replayed correction is outside the 250 ms hold gate"
    if abs(packet.joint_sync_gap_ns) > 10_000_000:
        return "replayed joint synchronization exceeds 10 ms"
    if previous_packet is not None:
        if packet.source_epoch != previous_packet.source_epoch:
            return "source epoch changed inside candidate window"
        if packet.sequence <= previous_packet.sequence:
            return "replay requires a distinct increasing source sequence"
    return None


def _empty_report(
    *,
    stats: _SourceStats,
    duplicate_sequences: int,
    expected_calibration_digest: bytes,
    expected_joint_mapping_digest: bytes,
    rejected: Sequence[RejectedCandidateInterval],
) -> ReplayConversionReport:
    return ReplayConversionReport(
        samples=0,
        source_records_read=stats.records_read,
        duplicate_sequences=duplicate_sequences,
        root_age_p95_ms=0.0,
        correction_age_p95_ms=0.0,
        joint_sync_abs_p95_ms=0.0,
        joint_sync_abs_max_ms=0.0,
        calibration_digest_sha256=expected_calibration_digest.hex(),
        joint_mapping_digest_sha256=expected_joint_mapping_digest.hex(),
        selected_query_start_ns=None,
        selected_query_end_ns=None,
        selected_source_publish_start_ns=None,
        selected_source_publish_end_ns=None,
        rejected_candidate_intervals=tuple(rejected),
    )


def write_policy_state_replay(
    source_path: Path | str,
    output_path: Path | str,
    *,
    samples: int,
    rate_hz: float,
    expected_calibration_digest: bytes,
    expected_joint_mapping_digest: bytes,
    warmup_seconds: float = 0.0,
) -> ReplayConversionReport:
    """Select the earliest complete strict replay window without bulk loading."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("rate_hz must be positive and finite")
    if not math.isfinite(warmup_seconds) or warmup_seconds < 0.0:
        raise ValueError("warmup_seconds must be non-negative and finite")
    for name, digest in (
        ("expected calibration digest", expected_calibration_digest),
        ("expected joint mapping digest", expected_joint_mapping_digest),
    ):
        if not isinstance(digest, bytes) or len(digest) != 32:
            raise ValueError(f"{name} must contain 32 raw bytes")
    if output_path.exists():
        raise PolicyStateTraceError(
            f"refusing to overwrite replay output: {output_path}"
        )

    period_ns = round(1_000_000_000.0 / rate_hz)
    if period_ns <= 0:
        raise ValueError("rate_hz produces a non-positive period")
    warmup_ns = round(warmup_seconds * 1_000_000_000.0)

    stats = _SourceStats()
    packet_iterator = _iter_policy_packets(
        source_path,
        expected_calibration_digest=expected_calibration_digest,
        expected_joint_mapping_digest=expected_joint_mapping_digest,
        stats=stats,
    )
    try:
        first_source_packet = next(packet_iterator)
    except StopIteration as error:
        report = _empty_report(
            stats=stats,
            duplicate_sequences=0,
            expected_calibration_digest=expected_calibration_digest,
            expected_joint_mapping_digest=expected_joint_mapping_digest,
            rejected=(),
        )
        raise PolicyStateTraceError(
            "source contains no qualifying policy-state packet",
            report=report,
        ) from error

    warmup_boundary_ns = first_source_packet.publish_time_ns + warmup_ns
    candidate = first_source_packet
    while candidate.publish_time_ns < warmup_boundary_ns:
        try:
            candidate = next(packet_iterator)
        except StopIteration as error:
            report = _empty_report(
                stats=stats,
                duplicate_sequences=0,
                expected_calibration_digest=expected_calibration_digest,
                expected_joint_mapping_digest=expected_joint_mapping_digest,
                rejected=(),
            )
            raise PolicyStateTraceError(
                "source has no qualifying packet after warmup",
                report=report,
            ) from error

    next_packet = next(packet_iterator, None)
    query_time_ns = candidate.publish_time_ns
    window: list[tuple[int, G1PolicyStatePacketV1]] = []
    rejected: list[RejectedCandidateInterval] = []
    duplicate_sequences = 0

    while True:
        while (
            next_packet is not None
            and next_packet.publish_time_ns <= query_time_ns
        ):
            candidate = next_packet
            next_packet = next(packet_iterator, None)

        previous_packet = window[-1][1] if window else None
        reason = _candidate_rejection(
            candidate,
            query_time_ns=query_time_ns,
            previous_packet=previous_packet,
        )
        if reason is None:
            window.append((query_time_ns, candidate))
            if len(window) == samples:
                break
        else:
            if "distinct increasing" in reason:
                duplicate_sequences += 1
            rejected.append(
                RejectedCandidateInterval(
                    query_start_ns=(
                        window[0][0] if window else query_time_ns
                    ),
                    query_end_ns=query_time_ns,
                    samples_before_rejection=len(window),
                    reason=reason,
                )
            )
            window.clear()

        if next_packet is None:
            break
        query_time_ns += period_ns

    if len(window) != samples:
        report = _empty_report(
            stats=stats,
            duplicate_sequences=duplicate_sequences,
            expected_calibration_digest=expected_calibration_digest,
            expected_joint_mapping_digest=expected_joint_mapping_digest,
            rejected=rejected,
        )
        raise PolicyStateTraceError(
            "source contains no qualifying complete replay window",
            report=report,
        )

    root_ages_ns = [
        query_ns - packet.estimate_time_ns for query_ns, packet in window
    ]
    correction_ages_ns = [
        query_ns - packet.correction_time_ns for query_ns, packet in window
    ]
    joint_sync_abs_ns = [
        abs(packet.joint_sync_gap_ns) for _, packet in window
    ]
    report = ReplayConversionReport(
        samples=len(window),
        source_records_read=stats.records_read,
        duplicate_sequences=duplicate_sequences,
        root_age_p95_ms=_percentile_95(root_ages_ns) / 1_000_000.0,
        correction_age_p95_ms=(
            _percentile_95(correction_ages_ns) / 1_000_000.0
        ),
        joint_sync_abs_p95_ms=(
            _percentile_95(joint_sync_abs_ns) / 1_000_000.0
        ),
        joint_sync_abs_max_ms=max(joint_sync_abs_ns) / 1_000_000.0,
        calibration_digest_sha256=expected_calibration_digest.hex(),
        joint_mapping_digest_sha256=expected_joint_mapping_digest.hex(),
        selected_query_start_ns=window[0][0],
        selected_query_end_ns=window[-1][0],
        selected_source_publish_start_ns=window[0][1].publish_time_ns,
        selected_source_publish_end_ns=window[-1][1].publish_time_ns,
        rejected_candidate_intervals=tuple(rejected),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output:
        for tick, (query_ns, packet) in enumerate(window):
            output.write(
                json.dumps(
                    {
                        "schema": "g1_policy_state_replay_v1",
                        "tick": tick,
                        "query_time_ns": query_ns,
                        "sequence": packet.sequence,
                        "source_epoch": packet.source_epoch,
                        "estimate_time_ns": packet.estimate_time_ns,
                        "publish_time_ns": packet.publish_time_ns,
                        "correction_time_ns": packet.correction_time_ns,
                        "joint_time_ns": packet.joint_time_ns,
                        "payload_hex": serialize_policy_state_v1(packet).hex(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return report
