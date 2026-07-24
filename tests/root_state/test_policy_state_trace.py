from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from g1_root_state_bridge.joint_contract import CANONICAL_G1_JOINT_NAMES
from g1_root_state_bridge.joint_transport import canonical_joint_mapping_digest
from g1_root_state_bridge.policy_state_protocol import (
    serialize_policy_state_v1,
)
from g1_root_state_bridge.policy_state_trace import (
    PolicyStateTraceError,
    policy_state_from_status_record,
    write_policy_state_replay,
    write_simulation_pair,
)
from g1_root_state_bridge.protocol import (
    REQUIRED_HEALTH_FLAGS,
    RootStateHealth,
    RootStatePacketV2,
    serialize_root_state_v2,
)


BASE_NS = 1_700_000_000_000_000_000
CALIBRATION_DIGEST = b"c" * 32
JOINT_MAPPING_DIGEST = canonical_joint_mapping_digest()


def status_record(
    sequence: int,
    publish_offset_ms: int,
    *,
    estimate_offset_ms: int | None = None,
    correction_offset_ms: int = -100,
    health_flags: RootStateHealth = RootStateHealth(REQUIRED_HEALTH_FLAGS),
) -> dict[str, object]:
    publish_time_ns = BASE_NS + publish_offset_ms * 1_000_000
    estimate_time_ns = (
        publish_time_ns
        if estimate_offset_ms is None
        else BASE_NS + estimate_offset_ms * 1_000_000
    )
    root = RootStatePacketV2(
        sequence=sequence,
        source_epoch=3,
        estimate_time_ns=estimate_time_ns,
        publish_time_ns=publish_time_ns,
        correction_time_ns=BASE_NS + correction_offset_ms * 1_000_000,
        joint_time_ns=estimate_time_ns + 250_000,
        joint_sync_gap_ns=250_000,
        health_flags=health_flags,
        position=(0.0, 0.0, 0.72),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        covariance_diagonal=(0.01,) * 6,
        calibration_digest=CALIBRATION_DIGEST,
    )
    return {
        "event": "packet_published",
        "kind": "packet",
        "payload_hex": serialize_root_state_v2(root).hex(),
        "strictly_valid": root.strictly_valid,
        "joint_names": list(CANONICAL_G1_JOINT_NAMES),
        "joint_position": [sequence / 100.0] * 29,
        "joint_velocity": [-sequence / 1000.0] * 29,
        "joint_mapping_digest_sha256": JOINT_MAPPING_DIGEST.hex(),
    }


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_status_record_conversion_preserves_root_and_exact_joint_evidence() -> None:
    record = status_record(4, 7)
    packet = policy_state_from_status_record(
        record,
        expected_calibration_digest=CALIBRATION_DIGEST,
        expected_joint_mapping_digest=JOINT_MAPPING_DIGEST,
    )
    assert packet.sequence == 4
    assert packet.publish_time_ns == BASE_NS + 7_000_000
    assert packet.joint_position == pytest.approx([0.04] * 29)
    assert packet.joint_velocity == pytest.approx([-0.004] * 29)
    assert packet.calibration_digest == CALIBRATION_DIGEST
    assert packet.joint_mapping_digest == JOINT_MAPPING_DIGEST


def test_new_status_record_requires_recorded_policy_packet_byte_parity() -> None:
    record = status_record(5, 8)
    packet = policy_state_from_status_record(
        record,
        expected_calibration_digest=CALIBRATION_DIGEST,
        expected_joint_mapping_digest=JOINT_MAPPING_DIGEST,
    )
    record["policy_state_payload_hex"] = serialize_policy_state_v1(packet).hex()
    recorded = policy_state_from_status_record(
        record,
        expected_calibration_digest=CALIBRATION_DIGEST,
        expected_joint_mapping_digest=JOINT_MAPPING_DIGEST,
    )
    assert serialize_policy_state_v1(recorded) == serialize_policy_state_v1(
        packet
    )

    altered = replace(
        packet,
        joint_position=(0.0,) * 29,
    )
    record["policy_state_payload_hex"] = serialize_policy_state_v1(altered).hex()
    with pytest.raises(PolicyStateTraceError, match="differs"):
        policy_state_from_status_record(
            record,
            expected_calibration_digest=CALIBRATION_DIGEST,
            expected_joint_mapping_digest=JOINT_MAPPING_DIGEST,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"joint_names": list(reversed(CANONICAL_G1_JOINT_NAMES))}, "canonical"),
        ({"joint_mapping_digest_sha256": "00" * 32}, "mapping digest"),
        ({"joint_position": [0.0] * 28}, "29 finite"),
        ({"joint_velocity": [0.0] * 28 + [math.nan]}, "29 finite"),
    ],
)
def test_status_record_rejects_invalid_joint_evidence(
    mutation: dict[str, object],
    message: str,
) -> None:
    record = {**status_record(1, 0), **mutation}
    with pytest.raises(PolicyStateTraceError, match=message):
        policy_state_from_status_record(
            record,
            expected_calibration_digest=CALIBRATION_DIGEST,
            expected_joint_mapping_digest=JOINT_MAPPING_DIGEST,
        )


def test_replay_conversion_selects_latest_published_packet_at_50hz(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bridge.jsonl"
    output = tmp_path / "policy.jsonl"
    write_records(
        source,
        [
            status_record(sequence, publish_offset_ms)
            for sequence, publish_offset_ms in ((1, 0), (2, 5), (3, 15))
        ],
    )

    report = write_policy_state_replay(
        source,
        output,
        samples=2,
        rate_hz=50.0,
        expected_calibration_digest=CALIBRATION_DIGEST,
        expected_joint_mapping_digest=JOINT_MAPPING_DIGEST,
    )

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["sequence"] for record in records] == [1, 3]
    assert all(
        record["query_time_ns"] >= record["publish_time_ns"]
        for record in records
    )
    assert report.samples == 2
    assert report.duplicate_sequences == 0
    assert report.source_records_read == 3


def test_replay_skips_non_packet_records_and_honors_source_warmup(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bridge.jsonl"
    output = tmp_path / "policy.jsonl"
    records = [{"event": "health_accepted"}]
    records.extend(
        status_record(index + 1, index * 20)
        for index in range(5)
    )
    write_records(source, records)

    report = write_policy_state_replay(
        source,
        output,
        samples=2,
        rate_hz=50.0,
        warmup_seconds=0.04,
        expected_calibration_digest=CALIBRATION_DIGEST,
        expected_joint_mapping_digest=JOINT_MAPPING_DIGEST,
    )

    replay = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["sequence"] for record in replay] == [3, 4]
    assert report.selected_query_start_ns == BASE_NS + 40_000_000


def test_replay_rejects_transient_unhealthy_candidate_then_recovers(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bridge.jsonl"
    output = tmp_path / "policy.jsonl"
    incomplete_health = RootStateHealth(
        int(REQUIRED_HEALTH_FLAGS)
        & ~int(RootStateHealth.CORRECTION_FRESH)
    )
    write_records(
        source,
        [
            status_record(1, 0, health_flags=incomplete_health),
            status_record(2, 20),
            status_record(3, 40),
        ],
    )

    report = write_policy_state_replay(
        source,
        output,
        samples=2,
        rate_hz=50.0,
        expected_calibration_digest=CALIBRATION_DIGEST,
        expected_joint_mapping_digest=JOINT_MAPPING_DIGEST,
    )

    replay = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["sequence"] for record in replay] == [2, 3]
    assert report.rejected_candidate_intervals[0].reason == (
        "policy health flags are incomplete"
    )


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                status_record(1, 0),
                status_record(2, 20, estimate_offset_ms=0),
            ],
            "qualifying",
        ),
        (
            [
                status_record(1, 0, correction_offset_ms=-260),
                status_record(2, 20, correction_offset_ms=-260),
            ],
            "qualifying",
        ),
        (
            [
                status_record(1, 20),
                status_record(2, 0),
            ],
            "publish times",
        ),
        (
            [
                status_record(1, 0),
            ],
            "qualifying",
        ),
    ],
)
def test_replay_fails_closed_without_a_complete_causal_fresh_window(
    tmp_path: Path,
    records: list[dict[str, object]],
    message: str,
) -> None:
    source = tmp_path / "bridge.jsonl"
    output = tmp_path / "policy.jsonl"
    write_records(source, records)

    with pytest.raises(PolicyStateTraceError, match=message):
        write_policy_state_replay(
            source,
            output,
            samples=2,
            rate_hz=50.0,
            expected_calibration_digest=CALIBRATION_DIGEST,
            expected_joint_mapping_digest=JOINT_MAPPING_DIGEST,
        )
    assert not output.exists()


def test_simulation_pair_contains_independent_direct_and_wire_state(
    tmp_path: Path,
) -> None:
    output = tmp_path / "pair.jsonl"

    write_simulation_pair(
        output,
        calibration_digest=CALIBRATION_DIGEST,
        joint_mapping_digest=JOINT_MAPPING_DIGEST,
    )

    records = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert [record["sequence"] for record in records] == [101, 102]
    assert all(record["schema"] == "g1_policy_state_replay_v1" for record in records)
    assert all("payload_hex" in record for record in records)
    assert all("direct_state" in record for record in records)
    for record in records:
        direct = record["direct_state"]
        assert direct["sequence"] == record["sequence"]
        assert direct["estimate_time_ns"] == record["estimate_time_ns"]
        assert len(direct["joint_position"]) == 29
        assert len(bytes.fromhex(record["payload_hex"])) == 440
