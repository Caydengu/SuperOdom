"""CRC-protected wire transport for source-timestamped G1 pelvis IMU data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import math
import struct
import zlib

from g1_root_state_bridge.root_imu_contract import TimedRootImuSample


ROOT_IMU_PACKET_MAGIC = b"HSIMU001"
ROOT_IMU_PACKET_BODY_STRUCT = struct.Struct("<8sQQqII10f")
ROOT_IMU_PACKET_CRC_STRUCT = struct.Struct("<I")
ROOT_IMU_PACKET_NUM_BYTES = (
    ROOT_IMU_PACKET_BODY_STRUCT.size + ROOT_IMU_PACKET_CRC_STRUCT.size
)


class RootImuPacketError(ValueError):
    """A root-IMU datagram violates the versioned wire contract."""


class RootImuHealth(IntFlag):
    SOURCE_TYPED = 1 << 0
    FINITE = 1 << 1
    QUATERNION_VALID = 1 << 2
    CLOCK_VALID = 1 << 3


REQUIRED_ROOT_IMU_HEALTH_FLAGS = (
    RootImuHealth.SOURCE_TYPED
    | RootImuHealth.FINITE
    | RootImuHealth.QUATERNION_VALID
    | RootImuHealth.CLOCK_VALID
)


@dataclass(frozen=True)
class RootImuPacketV1:
    source_epoch: int
    sequence: int
    stamp_ns: int
    source_tick: int
    health_flags: RootImuHealth
    quaternion_wxyz: tuple[float, float, float, float]
    angular_velocity: tuple[float, float, float]
    linear_acceleration: tuple[float, float, float]

    @property
    def strictly_valid(self) -> bool:
        return (
            self.health_flags & REQUIRED_ROOT_IMU_HEALTH_FLAGS
        ) == REQUIRED_ROOT_IMU_HEALTH_FLAGS

    def to_timed_sample(self, receipt_ns: int) -> TimedRootImuSample:
        return TimedRootImuSample(
            stamp_ns=self.stamp_ns,
            receipt_ns=receipt_ns,
            quaternion_wxyz=self.quaternion_wxyz,
            angular_velocity=self.angular_velocity,
            linear_acceleration=self.linear_acceleration,
            sequence=self.sequence,
            source_epoch=self.source_epoch,
        )


def _validate_packet(packet: RootImuPacketV1) -> None:
    for name, value in (
        ("source_epoch", packet.source_epoch),
        ("sequence", packet.sequence),
    ):
        if not isinstance(value, int) or not 0 <= value <= (1 << 64) - 1:
            raise RootImuPacketError(f"{name} must be an unsigned 64-bit integer")
    if not isinstance(packet.stamp_ns, int) or packet.stamp_ns < 0:
        raise RootImuPacketError("stamp_ns must be a non-negative integer")
    if not isinstance(packet.source_tick, int) or not 0 <= packet.source_tick <= (1 << 32) - 1:
        raise RootImuPacketError("source_tick must be an unsigned 32-bit integer")
    if not 0 <= int(packet.health_flags) <= (1 << 32) - 1:
        raise RootImuPacketError("health_flags must fit uint32")
    values = (
        *packet.quaternion_wxyz,
        *packet.angular_velocity,
        *packet.linear_acceleration,
    )
    if len(packet.quaternion_wxyz) != 4 or len(values) != 10:
        raise RootImuPacketError("root IMU packet vector lengths are invalid")
    if not all(math.isfinite(value) for value in values):
        raise RootImuPacketError("root IMU values must be finite")
    norm = math.sqrt(sum(value * value for value in packet.quaternion_wxyz))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise RootImuPacketError("quaternion_wxyz must be normalized")


def serialize_root_imu_packet(packet: RootImuPacketV1) -> bytes:
    _validate_packet(packet)
    body = ROOT_IMU_PACKET_BODY_STRUCT.pack(
        ROOT_IMU_PACKET_MAGIC,
        packet.source_epoch,
        packet.sequence,
        packet.stamp_ns,
        packet.source_tick,
        int(packet.health_flags),
        *packet.quaternion_wxyz,
        *packet.angular_velocity,
        *packet.linear_acceleration,
    )
    return body + ROOT_IMU_PACKET_CRC_STRUCT.pack(zlib.crc32(body))


def deserialize_root_imu_packet(data: bytes) -> RootImuPacketV1:
    if len(data) != ROOT_IMU_PACKET_NUM_BYTES:
        raise RootImuPacketError(
            f"root IMU packet length must be {ROOT_IMU_PACKET_NUM_BYTES}, got {len(data)}"
        )
    body = data[: ROOT_IMU_PACKET_BODY_STRUCT.size]
    expected_crc = ROOT_IMU_PACKET_CRC_STRUCT.unpack_from(data, len(body))[0]
    if zlib.crc32(body) != expected_crc:
        raise RootImuPacketError("root IMU packet CRC mismatch")
    values = ROOT_IMU_PACKET_BODY_STRUCT.unpack(body)
    if values[0] != ROOT_IMU_PACKET_MAGIC:
        raise RootImuPacketError("root IMU packet magic mismatch")
    packet = RootImuPacketV1(
        source_epoch=values[1],
        sequence=values[2],
        stamp_ns=values[3],
        source_tick=values[4],
        health_flags=RootImuHealth(values[5]),
        quaternion_wxyz=values[6:10],
        angular_velocity=values[10:13],
        linear_acceleration=values[13:16],
    )
    _validate_packet(packet)
    return packet


class RootImuPacketDecoder:
    """Strict freshness and ordering gate for root-IMU datagrams."""

    def __init__(self, *, max_age_ms: float):
        if max_age_ms <= 0.0:
            raise ValueError("max_age_ms must be positive")
        self.max_age_ns = int(max_age_ms * 1_000_000)
        self._last_source_epoch: int | None = None
        self._last_sequence: int | None = None
        self.last_rejection_reason: str | None = None

    def _reject(self, reason: str) -> None:
        self.last_rejection_reason = reason
        raise RootImuPacketError(reason)

    def decode(self, data: bytes, *, receipt_ns: int) -> TimedRootImuSample:
        try:
            packet = deserialize_root_imu_packet(data)
        except RootImuPacketError:
            self.last_rejection_reason = "root_imu_packet_malformed"
            raise
        if not packet.strictly_valid:
            self._reject("root_imu_health_flags_missing")
        age_ns = receipt_ns - packet.stamp_ns
        if age_ns < 0:
            self._reject("root_imu_clock_invalid")
        if age_ns > self.max_age_ns:
            self._reject("root_imu_stale")
        if self._last_source_epoch is not None:
            if packet.source_epoch < self._last_source_epoch:
                self._reject("root_imu_source_epoch_not_increasing")
            if packet.source_epoch == self._last_source_epoch:
                assert self._last_sequence is not None
                if packet.sequence <= self._last_sequence:
                    self._reject("root_imu_sequence_not_increasing")
        sample = packet.to_timed_sample(receipt_ns)
        self._last_source_epoch = packet.source_epoch
        self._last_sequence = packet.sequence
        self.last_rejection_reason = None
        return sample
