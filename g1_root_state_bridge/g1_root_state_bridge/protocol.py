"""Fixed wire contract for timestamped G1 pelvis state.

Version 2 is deliberately a fixed-size, little-endian packet so the Humble
producer and non-ROS Holosoma consumer can validate the same bytes without
sharing generated ROS types.

``correction_time_ns`` is the newest physical LiDAR-return time incorporated
by the latest state-estimator-confirmed correction.  It is deliberately not
the scan-start pose-reference time or the host application time; those remain
explicit on the ROS-side correction contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import math
import struct


ROOT_STATE_V2_MAGIC = b"HSROOT02"
ROOT_STATE_V2_STRUCT = struct.Struct("<8sQQqqqqqI19f32s")
ROOT_STATE_V2_NUM_BYTES = ROOT_STATE_V2_STRUCT.size


class RootStateProtocolError(ValueError):
    """The payload violates the versioned root-state wire contract."""


class RootStateHealth(IntFlag):
    """Independent evidence bits required by strict real deployment."""

    ESTIMATOR_HEALTHY = 1 << 0
    FINITE_POSE = 1 << 1
    JOINT_SYNC_VALID = 1 << 2
    CORRECTION_FRESH = 1 << 3
    CALIBRATION_VALID = 1 << 4
    CLOCK_VALID = 1 << 5
    ROOT_IMU_SYNC_VALID = 1 << 6
    ROOT_ORIENTATION_FUSED = 1 << 7


REQUIRED_HEALTH_FLAGS = (
    RootStateHealth.ESTIMATOR_HEALTHY
    | RootStateHealth.FINITE_POSE
    | RootStateHealth.JOINT_SYNC_VALID
    | RootStateHealth.CORRECTION_FRESH
    | RootStateHealth.CALIBRATION_VALID
    | RootStateHealth.CLOCK_VALID
)

REQUIRED_ROOT_FUSION_FLAGS = (
    REQUIRED_HEALTH_FLAGS
    | RootStateHealth.ROOT_IMU_SYNC_VALID
    | RootStateHealth.ROOT_ORIENTATION_FUSED
)


@dataclass(frozen=True)
class RootStatePacketV2:
    sequence: int
    source_epoch: int
    estimate_time_ns: int
    publish_time_ns: int
    correction_time_ns: int
    joint_time_ns: int
    joint_sync_gap_ns: int
    health_flags: RootStateHealth
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    covariance_diagonal: tuple[float, float, float, float, float, float]
    calibration_digest: bytes

    @property
    def strictly_valid(self) -> bool:
        return (self.health_flags & REQUIRED_HEALTH_FLAGS) == REQUIRED_HEALTH_FLAGS


def _validate_packet(packet: RootStatePacketV2) -> None:
    for name, value in (("sequence", packet.sequence), ("source_epoch", packet.source_epoch)):
        if not isinstance(value, int) or not 0 <= value <= (1 << 64) - 1:
            raise RootStateProtocolError(f"{name} must be an unsigned 64-bit integer")

    times = {
        "estimate_time_ns": packet.estimate_time_ns,
        "publish_time_ns": packet.publish_time_ns,
        "correction_time_ns": packet.correction_time_ns,
        "joint_time_ns": packet.joint_time_ns,
    }
    for name, value in times.items():
        if not isinstance(value, int) or value < 0:
            raise RootStateProtocolError(f"{name} must be a non-negative integer")
    if packet.publish_time_ns < packet.estimate_time_ns:
        raise RootStateProtocolError("publish time cannot predate estimate time")
    if packet.correction_time_ns > packet.estimate_time_ns:
        raise RootStateProtocolError("correction time cannot be in the estimate future")
    expected_gap = packet.joint_time_ns - packet.estimate_time_ns
    if packet.joint_sync_gap_ns != expected_gap:
        raise RootStateProtocolError(
            "joint_sync_gap_ns must equal joint_time_ns - estimate_time_ns"
        )

    health = int(packet.health_flags)
    if not 0 <= health <= (1 << 32) - 1:
        raise RootStateProtocolError("health_flags must fit uint32")

    vector_lengths = (
        ("position", packet.position, 3),
        ("quaternion_wxyz", packet.quaternion_wxyz, 4),
        ("linear_velocity", packet.linear_velocity, 3),
        ("angular_velocity", packet.angular_velocity, 3),
        ("covariance_diagonal", packet.covariance_diagonal, 6),
    )
    for name, values, expected_length in vector_lengths:
        if len(values) != expected_length:
            raise RootStateProtocolError(f"{name} must contain {expected_length} values")
        if not all(math.isfinite(value) for value in values):
            raise RootStateProtocolError(f"{name} values must be finite")

    quaternion_norm = math.sqrt(sum(value * value for value in packet.quaternion_wxyz))
    if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise RootStateProtocolError("quaternion_wxyz must be normalized")
    if any(value < 0.0 for value in packet.covariance_diagonal):
        raise RootStateProtocolError("covariance diagonal values must be non-negative")
    if not isinstance(packet.calibration_digest, bytes) or len(packet.calibration_digest) != 32:
        raise RootStateProtocolError("calibration digest must contain exactly 32 raw bytes")


def serialize_root_state_v2(packet: RootStatePacketV2) -> bytes:
    """Validate and serialize one fixed-size V2 packet."""
    _validate_packet(packet)
    return ROOT_STATE_V2_STRUCT.pack(
        ROOT_STATE_V2_MAGIC,
        packet.sequence,
        packet.source_epoch,
        packet.estimate_time_ns,
        packet.publish_time_ns,
        packet.correction_time_ns,
        packet.joint_time_ns,
        packet.joint_sync_gap_ns,
        int(packet.health_flags),
        *packet.position,
        *packet.quaternion_wxyz,
        *packet.linear_velocity,
        *packet.angular_velocity,
        *packet.covariance_diagonal,
        packet.calibration_digest,
    )


def deserialize_root_state_v2(data: bytes) -> RootStatePacketV2:
    """Deserialize and semantically validate one V2 packet."""
    if len(data) != ROOT_STATE_V2_NUM_BYTES:
        raise RootStateProtocolError(
            f"root-state V2 payload length must be {ROOT_STATE_V2_NUM_BYTES}, got {len(data)}"
        )
    values = ROOT_STATE_V2_STRUCT.unpack(data)
    if values[0] != ROOT_STATE_V2_MAGIC:
        raise RootStateProtocolError("root-state V2 magic mismatch")
    packet = RootStatePacketV2(
        sequence=values[1],
        source_epoch=values[2],
        estimate_time_ns=values[3],
        publish_time_ns=values[4],
        correction_time_ns=values[5],
        joint_time_ns=values[6],
        joint_sync_gap_ns=values[7],
        health_flags=RootStateHealth(values[8]),
        position=values[9:12],
        quaternion_wxyz=values[12:16],
        linear_velocity=values[16:19],
        angular_velocity=values[19:22],
        covariance_diagonal=values[22:28],
        calibration_digest=values[28],
    )
    _validate_packet(packet)
    return packet
