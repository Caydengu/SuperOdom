"""Atomic root-plus-joint packet for strict G1 policy shadowing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
import math
import struct

from g1_root_state_bridge.joint_contract import (
    CANONICAL_G1_JOINT_NAMES,
    TimedJointSample,
)
from g1_root_state_bridge.protocol import RootStatePacketV2


G1_POLICY_STATE_V1_MAGIC = b"HSPOLI01"
G1_POLICY_STATE_V1_STRUCT = struct.Struct("<8sQQqqqqqI77f32s32s")
G1_POLICY_STATE_V1_NUM_BYTES = G1_POLICY_STATE_V1_STRUCT.size


class PolicyStateProtocolError(ValueError):
    """The payload violates the G1 policy-state V1 contract."""


class PolicyStateHealth(IntFlag):
    """Independent evidence bits required by strict policy shadowing."""

    ESTIMATOR_HEALTHY = 1 << 0
    FINITE_POSE = 1 << 1
    JOINT_SYNC_VALID = 1 << 2
    CORRECTION_FRESH = 1 << 3
    CALIBRATION_VALID = 1 << 4
    CLOCK_VALID = 1 << 5
    JOINT_VALUES_FINITE = 1 << 6
    JOINT_MAPPING_VALID = 1 << 7


REQUIRED_POLICY_STATE_HEALTH_FLAGS = PolicyStateHealth(0xFF)


@dataclass(frozen=True)
class G1PolicyStatePacketV1:
    """One atomic pelvis and canonical 29-DOF joint-state sample."""

    sequence: int
    source_epoch: int
    estimate_time_ns: int
    publish_time_ns: int
    correction_time_ns: int
    joint_time_ns: int
    joint_sync_gap_ns: int
    health_flags: PolicyStateHealth
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    covariance_diagonal: tuple[float, float, float, float, float, float]
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    calibration_digest: bytes
    joint_mapping_digest: bytes

    @property
    def strictly_valid(self) -> bool:
        return (
            self.health_flags & REQUIRED_POLICY_STATE_HEALTH_FLAGS
        ) == REQUIRED_POLICY_STATE_HEALTH_FLAGS


def _validate_vector(
    name: str,
    values: tuple[float, ...],
    size: int,
) -> None:
    if len(values) != size:
        raise PolicyStateProtocolError(f"{name} must contain {size} values")
    if not all(math.isfinite(value) for value in values):
        raise PolicyStateProtocolError(f"{name} values must be finite")


def _validate_packet(packet: G1PolicyStatePacketV1) -> None:
    for name, value in (
        ("sequence", packet.sequence),
        ("source_epoch", packet.source_epoch),
    ):
        if not isinstance(value, int) or not 0 <= value <= (1 << 64) - 1:
            raise PolicyStateProtocolError(
                f"{name} must be an unsigned 64-bit integer"
            )

    for name, value in (
        ("estimate_time_ns", packet.estimate_time_ns),
        ("publish_time_ns", packet.publish_time_ns),
        ("correction_time_ns", packet.correction_time_ns),
        ("joint_time_ns", packet.joint_time_ns),
    ):
        if not isinstance(value, int) or value < 0:
            raise PolicyStateProtocolError(
                f"{name} must be a non-negative integer"
            )
    if packet.publish_time_ns < packet.estimate_time_ns:
        raise PolicyStateProtocolError("publish time cannot predate estimate time")
    if packet.correction_time_ns > packet.estimate_time_ns:
        raise PolicyStateProtocolError(
            "correction time cannot be in the estimate future"
        )
    if packet.joint_sync_gap_ns != (
        packet.joint_time_ns - packet.estimate_time_ns
    ):
        raise PolicyStateProtocolError(
            "joint_sync_gap_ns must equal joint_time_ns - estimate_time_ns"
        )

    if not 0 <= int(packet.health_flags) <= (1 << 32) - 1:
        raise PolicyStateProtocolError("health_flags must fit uint32")

    _validate_vector("position", packet.position, 3)
    _validate_vector("quaternion_wxyz", packet.quaternion_wxyz, 4)
    _validate_vector("linear_velocity", packet.linear_velocity, 3)
    _validate_vector("angular_velocity", packet.angular_velocity, 3)
    _validate_vector("covariance_diagonal", packet.covariance_diagonal, 6)
    _validate_vector("joint_position", packet.joint_position, 29)
    _validate_vector("joint_velocity", packet.joint_velocity, 29)

    quaternion_norm = math.sqrt(
        sum(value * value for value in packet.quaternion_wxyz)
    )
    if not math.isclose(quaternion_norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise PolicyStateProtocolError("quaternion_wxyz must be normalized")
    if any(value < 0.0 for value in packet.covariance_diagonal):
        raise PolicyStateProtocolError(
            "covariance diagonal values must be non-negative"
        )

    for name, digest in (
        ("calibration digest", packet.calibration_digest),
        ("joint mapping digest", packet.joint_mapping_digest),
    ):
        if not isinstance(digest, bytes) or len(digest) != 32:
            raise PolicyStateProtocolError(
                f"{name} must contain exactly 32 raw bytes"
            )


def build_policy_state_v1(
    root: RootStatePacketV2,
    joints: TimedJointSample,
    *,
    joint_mapping_digest: bytes,
) -> G1PolicyStatePacketV1:
    """Combine the root result with the exact joint sample used by its FK."""
    canonical = joints.canonicalized()
    if canonical.names != CANONICAL_G1_JOINT_NAMES:
        raise PolicyStateProtocolError("joint names are not canonical")
    if canonical.stamp_ns != root.joint_time_ns:
        raise PolicyStateProtocolError(
            "policy joints must be the exact root-FK joint sample"
        )

    packet = G1PolicyStatePacketV1(
        sequence=root.sequence,
        source_epoch=root.source_epoch,
        estimate_time_ns=root.estimate_time_ns,
        publish_time_ns=root.publish_time_ns,
        correction_time_ns=root.correction_time_ns,
        joint_time_ns=root.joint_time_ns,
        joint_sync_gap_ns=root.joint_sync_gap_ns,
        health_flags=(
            PolicyStateHealth(int(root.health_flags))
            | PolicyStateHealth.JOINT_VALUES_FINITE
            | PolicyStateHealth.JOINT_MAPPING_VALID
        ),
        position=root.position,
        quaternion_wxyz=root.quaternion_wxyz,
        linear_velocity=root.linear_velocity,
        angular_velocity=root.angular_velocity,
        covariance_diagonal=root.covariance_diagonal,
        joint_position=canonical.position,
        joint_velocity=canonical.velocity,
        calibration_digest=root.calibration_digest,
        joint_mapping_digest=joint_mapping_digest,
    )
    _validate_packet(packet)
    return packet


def serialize_policy_state_v1(packet: G1PolicyStatePacketV1) -> bytes:
    """Validate and serialize one fixed-size policy-state V1 packet."""
    _validate_packet(packet)
    return G1_POLICY_STATE_V1_STRUCT.pack(
        G1_POLICY_STATE_V1_MAGIC,
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
        *packet.joint_position,
        *packet.joint_velocity,
        packet.calibration_digest,
        packet.joint_mapping_digest,
    )


def deserialize_policy_state_v1(data: bytes) -> G1PolicyStatePacketV1:
    """Deserialize and semantically validate one policy-state V1 packet."""
    if len(data) != G1_POLICY_STATE_V1_NUM_BYTES:
        raise PolicyStateProtocolError(
            "policy-state V1 payload length must be "
            f"{G1_POLICY_STATE_V1_NUM_BYTES}, got {len(data)}"
        )

    values = G1_POLICY_STATE_V1_STRUCT.unpack(data)
    if values[0] != G1_POLICY_STATE_V1_MAGIC:
        raise PolicyStateProtocolError("policy-state V1 magic mismatch")

    packet = G1PolicyStatePacketV1(
        sequence=values[1],
        source_epoch=values[2],
        estimate_time_ns=values[3],
        publish_time_ns=values[4],
        correction_time_ns=values[5],
        joint_time_ns=values[6],
        joint_sync_gap_ns=values[7],
        health_flags=PolicyStateHealth(values[8]),
        position=values[9:12],
        quaternion_wxyz=values[12:16],
        linear_velocity=values[16:19],
        angular_velocity=values[19:22],
        covariance_diagonal=values[22:28],
        joint_position=values[28:57],
        joint_velocity=values[57:86],
        calibration_digest=values[86],
        joint_mapping_digest=values[87],
    )
    _validate_packet(packet)
    return packet
