"""Versioned transport for typed G1 joint samples stamped on the robot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from hashlib import sha256
import math
import struct
import zlib

from g1_root_state_bridge.joint_contract import (
    CANONICAL_G1_JOINT_NAMES,
    TimedJointSample,
)


JOINT_PACKET_MAGIC = b"HSJNT001"
JOINT_PACKET_BODY_STRUCT = struct.Struct("<8sQQqII58f32s")
JOINT_PACKET_CRC_STRUCT = struct.Struct("<I")
JOINT_PACKET_NUM_BYTES = JOINT_PACKET_BODY_STRUCT.size + JOINT_PACKET_CRC_STRUCT.size


class JointPacketError(ValueError):
    """A G1 joint datagram violates the frozen wire contract."""


class JointHealth(IntFlag):
    SOURCE_TYPED = 1 << 0
    FINITE = 1 << 1
    MOTOR_COUNT_VALID = 1 << 2
    CLOCK_VALID = 1 << 3


REQUIRED_JOINT_HEALTH_FLAGS = (
    JointHealth.SOURCE_TYPED
    | JointHealth.FINITE
    | JointHealth.MOTOR_COUNT_VALID
    | JointHealth.CLOCK_VALID
)


def canonical_joint_mapping_digest() -> bytes:
    """Hash the semantic mapping from Unitree motor indices 0..28 to names."""
    contract = "g1-29dof-unitree-motor-index-v1\0" + "\n".join(
        CANONICAL_G1_JOINT_NAMES
    )
    return sha256(contract.encode("utf-8")).digest()


@dataclass(frozen=True)
class JointPacketV1:
    source_epoch: int
    sequence: int
    stamp_ns: int
    source_tick: int
    health_flags: JointHealth
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    mapping_digest: bytes

    @property
    def strictly_valid(self) -> bool:
        return (
            self.health_flags & REQUIRED_JOINT_HEALTH_FLAGS
        ) == REQUIRED_JOINT_HEALTH_FLAGS

    def to_timed_sample(self, receipt_ns: int) -> TimedJointSample:
        return TimedJointSample(
            stamp_ns=self.stamp_ns,
            receipt_ns=receipt_ns,
            names=CANONICAL_G1_JOINT_NAMES,
            position=self.position,
            velocity=self.velocity,
            sequence=self.sequence,
            source_epoch=self.source_epoch,
        )


def _validate_packet(packet: JointPacketV1) -> None:
    for name, value in (
        ("source_epoch", packet.source_epoch),
        ("sequence", packet.sequence),
    ):
        if not isinstance(value, int) or not 0 <= value <= (1 << 64) - 1:
            raise JointPacketError(f"{name} must be an unsigned 64-bit integer")
    if not isinstance(packet.stamp_ns, int) or packet.stamp_ns < 0:
        raise JointPacketError("stamp_ns must be a non-negative integer")
    if not isinstance(packet.source_tick, int) or not 0 <= packet.source_tick <= (1 << 32) - 1:
        raise JointPacketError("source_tick must be an unsigned 32-bit integer")
    health = int(packet.health_flags)
    if not 0 <= health <= (1 << 32) - 1:
        raise JointPacketError("health_flags must fit uint32")
    if len(packet.position) != 29 or len(packet.velocity) != 29:
        raise JointPacketError("position and velocity must each contain 29 values")
    if not all(math.isfinite(value) for value in (*packet.position, *packet.velocity)):
        raise JointPacketError("joint position and velocity values must be finite")
    if not isinstance(packet.mapping_digest, bytes) or len(packet.mapping_digest) != 32:
        raise JointPacketError("mapping digest must contain exactly 32 raw bytes")


def serialize_joint_packet(packet: JointPacketV1) -> bytes:
    _validate_packet(packet)
    body = JOINT_PACKET_BODY_STRUCT.pack(
        JOINT_PACKET_MAGIC,
        packet.source_epoch,
        packet.sequence,
        packet.stamp_ns,
        packet.source_tick,
        int(packet.health_flags),
        *packet.position,
        *packet.velocity,
        packet.mapping_digest,
    )
    return body + JOINT_PACKET_CRC_STRUCT.pack(zlib.crc32(body))


def deserialize_joint_packet(data: bytes) -> JointPacketV1:
    if len(data) != JOINT_PACKET_NUM_BYTES:
        raise JointPacketError(
            f"joint packet length must be {JOINT_PACKET_NUM_BYTES}, got {len(data)}"
        )
    if data[:8] != JOINT_PACKET_MAGIC:
        raise JointPacketError("joint packet magic mismatch")
    body = data[: JOINT_PACKET_BODY_STRUCT.size]
    expected_crc = JOINT_PACKET_CRC_STRUCT.unpack_from(data, len(body))[0]
    actual_crc = zlib.crc32(body)
    if actual_crc != expected_crc:
        raise JointPacketError(
            f"joint packet CRC mismatch: expected {expected_crc:#010x}, got {actual_crc:#010x}"
        )
    values = JOINT_PACKET_BODY_STRUCT.unpack(body)
    packet = JointPacketV1(
        source_epoch=values[1],
        sequence=values[2],
        stamp_ns=values[3],
        source_tick=values[4],
        health_flags=JointHealth(values[5]),
        position=values[6:35],
        velocity=values[35:64],
        mapping_digest=values[64],
    )
    _validate_packet(packet)
    return packet


class JointPacketDecoder:
    """Strict receiver-side freshness, provenance, and ordering gate."""

    def __init__(self, *, allowed_mapping_digests: set[bytes], max_age_ms: float):
        if max_age_ms <= 0.0:
            raise ValueError("max_age_ms must be positive")
        if not allowed_mapping_digests:
            raise ValueError("at least one joint mapping digest must be allowlisted")
        if any(len(digest) != 32 for digest in allowed_mapping_digests):
            raise ValueError("joint mapping digests must contain 32 raw bytes")
        self.allowed_mapping_digests = frozenset(allowed_mapping_digests)
        self.max_age_ns = int(max_age_ms * 1_000_000)
        self._last_source_epoch: int | None = None
        self._last_sequence: int | None = None
        self.last_rejection_reason: str | None = None

    def _reject(self, reason: str) -> None:
        self.last_rejection_reason = reason
        raise JointPacketError(reason)

    def decode(self, data: bytes, *, receipt_ns: int) -> TimedJointSample:
        try:
            packet = deserialize_joint_packet(data)
        except JointPacketError:
            self.last_rejection_reason = "joint_packet_malformed"
            raise
        if not packet.strictly_valid:
            self._reject("joint_health_flags_missing")
        if packet.mapping_digest not in self.allowed_mapping_digests:
            self._reject("joint_mapping_not_allowed")
        age_ns = receipt_ns - packet.stamp_ns
        if age_ns < 0:
            self._reject("joint_clock_invalid")
        if age_ns > self.max_age_ns:
            self._reject("joint_state_stale")
        if self._last_source_epoch is not None:
            if packet.source_epoch < self._last_source_epoch:
                self._reject("joint_source_epoch_not_increasing")
            if packet.source_epoch == self._last_source_epoch:
                assert self._last_sequence is not None
                if packet.sequence <= self._last_sequence:
                    self._reject("joint_sequence_not_increasing")

        sample = packet.to_timed_sample(receipt_ns)
        self._last_source_epoch = packet.source_epoch
        self._last_sequence = packet.sequence
        self.last_rejection_reason = None
        return sample

    def decode_or_none(self, data: bytes, *, receipt_ns: int) -> TimedJointSample | None:
        try:
            return self.decode(data, receipt_ns=receipt_ns)
        except JointPacketError:
            return None
