from __future__ import annotations

from dataclasses import replace
import math

import pytest

from g1_root_state_bridge.joint_contract import CANONICAL_G1_JOINT_NAMES
from g1_root_state_bridge.joint_transport import (
    REQUIRED_JOINT_HEALTH_FLAGS,
    JOINT_PACKET_MAGIC,
    JOINT_PACKET_NUM_BYTES,
    JointHealth,
    JointPacketDecoder,
    JointPacketError,
    JointPacketV1,
    canonical_joint_mapping_digest,
    deserialize_joint_packet,
    serialize_joint_packet,
)


def packet(**mutations: object) -> JointPacketV1:
    base = JointPacketV1(
        source_epoch=1_000,
        sequence=42,
        stamp_ns=10_000_000_000,
        source_tick=123_456,
        health_flags=REQUIRED_JOINT_HEALTH_FLAGS,
        position=tuple(index + 0.25 for index in range(29)),
        velocity=tuple(-(index + 0.5) for index in range(29)),
        mapping_digest=canonical_joint_mapping_digest(),
    )
    return replace(base, **mutations)


def test_joint_packet_round_trip_is_fixed_and_crc_protected() -> None:
    encoded = serialize_joint_packet(packet())
    decoded = deserialize_joint_packet(encoded)

    assert JOINT_PACKET_MAGIC == b"HSJNT001"
    assert len(encoded) == JOINT_PACKET_NUM_BYTES == 308
    assert decoded.source_epoch == 1_000
    assert decoded.sequence == 42
    assert decoded.stamp_ns == 10_000_000_000
    assert decoded.source_tick == 123_456
    assert decoded.health_flags == REQUIRED_JOINT_HEALTH_FLAGS
    assert decoded.position[12:15] == (12.25, 13.25, 14.25)
    assert decoded.mapping_digest == canonical_joint_mapping_digest()
    assert decoded.strictly_valid


def test_joint_packet_converts_to_name_bound_timed_sample() -> None:
    sample = packet().to_timed_sample(receipt_ns=10_002_000_000)
    assert sample.names == CANONICAL_G1_JOINT_NAMES
    assert sample.source_epoch == 1_000
    assert sample.sequence == 42
    assert sample.stamp_ns == 10_000_000_000
    assert sample.receipt_ns == 10_002_000_000
    assert sample.position[12:15] == (12.25, 13.25, 14.25)


def test_joint_packet_rejects_corruption_wrong_magic_and_length() -> None:
    encoded = bytearray(serialize_joint_packet(packet()))
    encoded[100] ^= 0x01
    with pytest.raises(JointPacketError, match="CRC"):
        deserialize_joint_packet(bytes(encoded))
    with pytest.raises(JointPacketError, match="length"):
        deserialize_joint_packet(bytes(encoded[:-1]))

    wrong_magic = bytearray(serialize_joint_packet(packet()))
    wrong_magic[:8] = b"BADMAGIC"
    with pytest.raises(JointPacketError, match="magic"):
        deserialize_joint_packet(bytes(wrong_magic))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"source_epoch": -1}, "source_epoch"),
        ({"sequence": -1}, "sequence"),
        ({"stamp_ns": -1}, "stamp_ns"),
        ({"source_tick": -1}, "source_tick"),
        ({"position": (math.nan,) + (0.0,) * 28}, "finite"),
        ({"velocity": (0.0,) * 28}, "29"),
        ({"mapping_digest": b"short"}, "digest"),
    ],
)
def test_joint_packet_rejects_invalid_semantics(mutation: dict[str, object], match: str) -> None:
    with pytest.raises(JointPacketError, match=match):
        serialize_joint_packet(packet(**mutation))


def test_strict_decoder_rejects_stale_unhealthy_unknown_and_nonmonotonic_packets() -> None:
    digest = canonical_joint_mapping_digest()
    decoder = JointPacketDecoder(
        allowed_mapping_digests={digest},
        max_age_ms=10.0,
    )

    accepted = decoder.decode(
        serialize_joint_packet(packet()),
        receipt_ns=10_005_000_000,
    )
    assert accepted.sequence == 42

    assert decoder.decode_or_none(
        serialize_joint_packet(packet(sequence=43, stamp_ns=9_980_000_000)),
        receipt_ns=10_005_000_000,
    ) is None
    assert decoder.last_rejection_reason == "joint_state_stale"

    assert decoder.decode_or_none(
        serialize_joint_packet(
            packet(sequence=43, health_flags=REQUIRED_JOINT_HEALTH_FLAGS & ~JointHealth.CLOCK_VALID)
        ),
        receipt_ns=10_005_000_000,
    ) is None
    assert decoder.last_rejection_reason == "joint_health_flags_missing"

    assert decoder.decode_or_none(
        serialize_joint_packet(packet(sequence=43, mapping_digest=b"x" * 32)),
        receipt_ns=10_005_000_000,
    ) is None
    assert decoder.last_rejection_reason == "joint_mapping_not_allowed"

    assert decoder.decode_or_none(
        serialize_joint_packet(packet(sequence=42)),
        receipt_ns=10_005_000_000,
    ) is None
    assert decoder.last_rejection_reason == "joint_sequence_not_increasing"

    restarted = decoder.decode(
        serialize_joint_packet(packet(source_epoch=1_001, sequence=3)),
        receipt_ns=10_005_000_000,
    )
    assert restarted.source_epoch == 1_001
