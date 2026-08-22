"""Typed, source-timestamped pelvis IMU samples for G1 root fusion."""

from __future__ import annotations

from dataclasses import dataclass
import math


class RootImuContractError(ValueError):
    """A pelvis IMU sample violates the frozen measurement contract."""


@dataclass(frozen=True)
class TimedRootImuSample:
    """One pelvis-frame IMU sample stamped in the shared system-clock domain.

    ``quaternion_wxyz`` is the Unitree root IMU attitude quaternion.  Its yaw
    origin is calibrated against local LIO at fusion initialization; roll and
    pitch are gravity referenced.  Angular velocity and linear acceleration
    are expressed in the pelvis IMU frame.
    """

    stamp_ns: int
    receipt_ns: int
    quaternion_wxyz: tuple[float, float, float, float]
    angular_velocity: tuple[float, float, float]
    linear_acceleration: tuple[float, float, float]
    sequence: int
    source_epoch: int

    def __post_init__(self) -> None:
        if not isinstance(self.stamp_ns, int) or self.stamp_ns < 0:
            raise RootImuContractError("stamp_ns must be a non-negative integer")
        if not isinstance(self.receipt_ns, int) or self.receipt_ns < self.stamp_ns:
            raise RootImuContractError(
                "receipt_ns must be an integer no earlier than stamp_ns"
            )
        for name, value in (
            ("sequence", self.sequence),
            ("source_epoch", self.source_epoch),
        ):
            if not isinstance(value, int) or not 0 <= value <= (1 << 64) - 1:
                raise RootImuContractError(f"{name} must be an unsigned 64-bit integer")
        groups = (
            ("quaternion_wxyz", self.quaternion_wxyz, 4),
            ("angular_velocity", self.angular_velocity, 3),
            ("linear_acceleration", self.linear_acceleration, 3),
        )
        for name, values, expected in groups:
            if len(values) != expected or not all(math.isfinite(v) for v in values):
                raise RootImuContractError(
                    f"{name} must contain {expected} finite values"
                )
        norm = math.sqrt(sum(v * v for v in self.quaternion_wxyz))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
            raise RootImuContractError("quaternion_wxyz must be normalized")
