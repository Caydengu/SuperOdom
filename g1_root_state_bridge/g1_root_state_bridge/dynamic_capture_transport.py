"""Versioned, structurally read-only G1 low-state capture transport."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import math
import struct
import zlib

from g1_root_state_bridge.joint_transport import canonical_joint_mapping_digest


DYNAMIC_CAPTURE_PACKET_MAGIC = b"HSDYN001"
DYNAMIC_CAPTURE_PACKET_BODY = struct.Struct("<8sQQqII68f32s")
DYNAMIC_CAPTURE_PACKET_CRC = struct.Struct("<I")
DYNAMIC_CAPTURE_PACKET_NUM_BYTES = (
    DYNAMIC_CAPTURE_PACKET_BODY.size + DYNAMIC_CAPTURE_PACKET_CRC.size
)


class DynamicCapturePacketError(ValueError):
    """A source or recorded dynamic-capture packet violates its contract."""


class DynamicCaptureHealth(IntFlag):
    SOURCE_TYPED = 1 << 0
    FINITE = 1 << 1
    MOTOR_COUNT_VALID = 1 << 2
    IMU_VALID = 1 << 3
    CLOCK_VALID = 1 << 4


REQUIRED_DYNAMIC_CAPTURE_HEALTH = (
    DynamicCaptureHealth.SOURCE_TYPED
    | DynamicCaptureHealth.FINITE
    | DynamicCaptureHealth.MOTOR_COUNT_VALID
    | DynamicCaptureHealth.IMU_VALID
    | DynamicCaptureHealth.CLOCK_VALID
)


@dataclass(frozen=True)
class DynamicCapturePacketV1:
    source_epoch: int
    sequence: int
    robot_stamp_ns: int
    source_tick: int
    health_flags: DynamicCaptureHealth
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    imu_quaternion_wxyz: tuple[float, ...]
    imu_gyroscope: tuple[float, ...]
    imu_accelerometer: tuple[float, ...]
    mapping_digest: bytes

    @property
    def strictly_valid(self) -> bool:
        return (
            self.health_flags & REQUIRED_DYNAMIC_CAPTURE_HEALTH
        ) == REQUIRED_DYNAMIC_CAPTURE_HEALTH


def _validate_packet(packet: DynamicCapturePacketV1) -> None:
    if not isinstance(packet, DynamicCapturePacketV1):
        raise DynamicCapturePacketError("packet must be a DynamicCapturePacketV1")
    for name, value in (
        ("source_epoch", packet.source_epoch),
        ("sequence", packet.sequence),
    ):
        if not isinstance(value, int) or not 0 <= value <= (1 << 64) - 1:
            raise DynamicCapturePacketError(
                f"{name} must be an unsigned 64-bit integer"
            )
    if (
        not isinstance(packet.robot_stamp_ns, int)
        or not 0 <= packet.robot_stamp_ns <= (1 << 63) - 1
    ):
        raise DynamicCapturePacketError(
            "robot_stamp_ns must be a non-negative signed 64-bit integer"
        )
    if (
        not isinstance(packet.source_tick, int)
        or not 0 <= packet.source_tick <= (1 << 32) - 1
    ):
        raise DynamicCapturePacketError(
            "source_tick must be an unsigned 32-bit integer"
        )
    health_value = int(packet.health_flags)
    if not 0 <= health_value <= (1 << 32) - 1:
        raise DynamicCapturePacketError("health_flags must fit uint32")
    if len(packet.joint_position) != 29 or len(packet.joint_velocity) != 29:
        raise DynamicCapturePacketError(
            "joint position and velocity must each contain exactly 29 values"
        )
    if len(packet.imu_quaternion_wxyz) != 4:
        raise DynamicCapturePacketError(
            "IMU quaternion must contain exactly four values"
        )
    if len(packet.imu_gyroscope) != 3:
        raise DynamicCapturePacketError(
            "IMU gyroscope must contain exactly three values"
        )
    if len(packet.imu_accelerometer) != 3:
        raise DynamicCapturePacketError(
            "IMU accelerometer must contain exactly three values"
        )
    numeric_values = (
        *packet.joint_position,
        *packet.joint_velocity,
        *packet.imu_quaternion_wxyz,
        *packet.imu_gyroscope,
        *packet.imu_accelerometer,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise DynamicCapturePacketError(
            "joint and IMU values must all be finite"
        )
    if not isinstance(packet.mapping_digest, bytes) or len(packet.mapping_digest) != 32:
        raise DynamicCapturePacketError(
            "mapping digest must contain exactly 32 raw bytes"
        )


def serialize_dynamic_capture_packet(packet: DynamicCapturePacketV1) -> bytes:
    """Serialize one source sample with a trailing CRC32."""

    _validate_packet(packet)
    body = DYNAMIC_CAPTURE_PACKET_BODY.pack(
        DYNAMIC_CAPTURE_PACKET_MAGIC,
        packet.source_epoch,
        packet.sequence,
        packet.robot_stamp_ns,
        packet.source_tick,
        int(packet.health_flags),
        *packet.joint_position,
        *packet.joint_velocity,
        *packet.imu_quaternion_wxyz,
        *packet.imu_gyroscope,
        *packet.imu_accelerometer,
        packet.mapping_digest,
    )
    return body + DYNAMIC_CAPTURE_PACKET_CRC.pack(zlib.crc32(body))


def deserialize_dynamic_capture_packet(data: bytes) -> DynamicCapturePacketV1:
    """Decode and validate one fixed-size source sample."""

    if len(data) != DYNAMIC_CAPTURE_PACKET_NUM_BYTES:
        raise DynamicCapturePacketError(
            "dynamic capture packet length must be "
            f"{DYNAMIC_CAPTURE_PACKET_NUM_BYTES}, got {len(data)}"
        )
    if data[:8] != DYNAMIC_CAPTURE_PACKET_MAGIC:
        raise DynamicCapturePacketError("dynamic capture packet magic mismatch")
    body = data[: DYNAMIC_CAPTURE_PACKET_BODY.size]
    expected_crc = DYNAMIC_CAPTURE_PACKET_CRC.unpack_from(data, len(body))[0]
    actual_crc = zlib.crc32(body)
    if actual_crc != expected_crc:
        raise DynamicCapturePacketError(
            "dynamic capture packet CRC mismatch: "
            f"expected {expected_crc:#010x}, got {actual_crc:#010x}"
        )
    values = DYNAMIC_CAPTURE_PACKET_BODY.unpack(body)
    float_values = values[6:74]
    packet = DynamicCapturePacketV1(
        source_epoch=values[1],
        sequence=values[2],
        robot_stamp_ns=values[3],
        source_tick=values[4],
        health_flags=DynamicCaptureHealth(values[5]),
        joint_position=tuple(float_values[:29]),
        joint_velocity=tuple(float_values[29:58]),
        imu_quaternion_wxyz=tuple(float_values[58:62]),
        imu_gyroscope=tuple(float_values[62:65]),
        imu_accelerometer=tuple(float_values[65:68]),
        mapping_digest=values[74],
    )
    _validate_packet(packet)
    return packet


def dynamic_capture_mapping_digest() -> bytes:
    """Expose the frozen joint ordering identity without duplicating it."""

    return canonical_joint_mapping_digest()
