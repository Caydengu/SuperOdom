"""Typed, name-bound G1 joint sample contract."""

from __future__ import annotations

from dataclasses import dataclass
import math


CANONICAL_G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


class JointContractError(ValueError):
    """A joint sample cannot be interpreted under the frozen G1 contract."""


@dataclass(frozen=True)
class TimedJointSample:
    """One 29-DOF sample with source and receiving-host timestamps.

    ``stamp_ns`` is the source timestamp expressed in the shared system-clock
    domain. For the current Unitree relay it is G1 callback receipt time,
    because the vendor LowState payload has no capture timestamp.
    ``receipt_ns`` is when the bridge host receives the relayed sample.
    """

    stamp_ns: int
    receipt_ns: int
    names: tuple[str, ...]
    position: tuple[float, ...]
    velocity: tuple[float, ...]
    sequence: int
    source_epoch: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.stamp_ns, int) or self.stamp_ns < 0:
            raise JointContractError("stamp_ns must be a non-negative integer")
        if not isinstance(self.receipt_ns, int) or self.receipt_ns < self.stamp_ns:
            raise JointContractError("receipt_ns must be an integer no earlier than stamp_ns")
        if not isinstance(self.sequence, int) or not 0 <= self.sequence <= (1 << 64) - 1:
            raise JointContractError("sequence must be an unsigned 64-bit integer")
        if not isinstance(self.source_epoch, int) or not 0 <= self.source_epoch <= (1 << 64) - 1:
            raise JointContractError("source_epoch must be an unsigned 64-bit integer")
        if len(self.names) != 29:
            raise JointContractError("joint sample must contain exactly 29 names")
        if len(set(self.names)) != len(self.names):
            raise JointContractError("joint names contain duplicates")
        if set(self.names) != set(CANONICAL_G1_JOINT_NAMES):
            missing = sorted(set(CANONICAL_G1_JOINT_NAMES) - set(self.names))
            extra = sorted(set(self.names) - set(CANONICAL_G1_JOINT_NAMES))
            raise JointContractError(f"joint names do not match G1 contract; missing={missing}, extra={extra}")
        if len(self.position) != len(self.names) or len(self.velocity) != len(self.names):
            raise JointContractError("position and velocity length must match names")
        if not all(math.isfinite(value) for value in (*self.position, *self.velocity)):
            raise JointContractError("joint position and velocity values must be finite")

    def canonicalized(self) -> "TimedJointSample":
        """Return the sample reordered by explicit joint name."""
        if self.names == CANONICAL_G1_JOINT_NAMES:
            return self
        index_by_name = {name: index for index, name in enumerate(self.names)}
        order = tuple(index_by_name[name] for name in CANONICAL_G1_JOINT_NAMES)
        return TimedJointSample(
            stamp_ns=self.stamp_ns,
            receipt_ns=self.receipt_ns,
            names=CANONICAL_G1_JOINT_NAMES,
            position=tuple(self.position[index] for index in order),
            velocity=tuple(self.velocity[index] for index in order),
            sequence=self.sequence,
            source_epoch=self.source_epoch,
        )
