"""Timestamp synchronization for pelvis IMU samples."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from g1_root_state_bridge.root_imu_contract import TimedRootImuSample


class RootImuSynchronizationError(ValueError):
    """No trustworthy pelvis IMU sample brackets the requested source time."""


@dataclass(frozen=True)
class RootImuMatch:
    sample: TimedRootImuSample
    maximum_bracket_gap_ns: int


def _slerp_wxyz(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        q = a + alpha * (b - a)
        return q / np.linalg.norm(q)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return (
        math.sin((1.0 - alpha) * theta) / sin_theta * a
        + math.sin(alpha * theta) / sin_theta * b
    )


class RootImuBuffer:
    def __init__(self, capacity: int = 4096):
        if capacity < 2:
            raise ValueError("root IMU buffer capacity must be at least two")
        self._samples: deque[TimedRootImuSample] = deque(maxlen=capacity)

    def clear(self) -> None:
        self._samples.clear()

    def append(self, sample: TimedRootImuSample) -> None:
        if self._samples:
            previous = self._samples[-1]
            if sample.source_epoch < previous.source_epoch:
                raise RootImuSynchronizationError("root IMU source epoch regressed")
            if sample.source_epoch == previous.source_epoch:
                if sample.sequence <= previous.sequence:
                    raise RootImuSynchronizationError("root IMU sequence did not increase")
                if sample.stamp_ns <= previous.stamp_ns:
                    raise RootImuSynchronizationError("root IMU source time did not increase")
            else:
                self._samples.clear()
        self._samples.append(sample)

    def synchronize(self, query_time_ns: int, *, max_gap_ns: int) -> RootImuMatch:
        if len(self._samples) < 2:
            raise RootImuSynchronizationError("fewer than two root IMU samples")
        before = None
        after = None
        for sample in self._samples:
            if sample.stamp_ns <= query_time_ns:
                before = sample
            if sample.stamp_ns >= query_time_ns:
                after = sample
                break
        if before is None or after is None:
            raise RootImuSynchronizationError("root IMU does not bracket query time")
        if before.source_epoch != after.source_epoch:
            raise RootImuSynchronizationError("root IMU bracket crosses source epoch")
        left_gap = query_time_ns - before.stamp_ns
        right_gap = after.stamp_ns - query_time_ns
        maximum_gap = max(left_gap, right_gap)
        if maximum_gap > max_gap_ns:
            raise RootImuSynchronizationError(
                f"root IMU bracket gap {maximum_gap} ns exceeds {max_gap_ns} ns"
            )
        span = after.stamp_ns - before.stamp_ns
        alpha = 0.0 if span == 0 else left_gap / span
        qa = np.asarray(before.quaternion_wxyz, dtype=np.float64)
        qb = np.asarray(after.quaternion_wxyz, dtype=np.float64)
        quaternion = _slerp_wxyz(qa, qb, alpha)
        angular = (
            (1.0 - alpha) * np.asarray(before.angular_velocity)
            + alpha * np.asarray(after.angular_velocity)
        )
        acceleration = (
            (1.0 - alpha) * np.asarray(before.linear_acceleration)
            + alpha * np.asarray(after.linear_acceleration)
        )
        return RootImuMatch(
            sample=TimedRootImuSample(
                stamp_ns=query_time_ns,
                receipt_ns=max(before.receipt_ns, after.receipt_ns),
                quaternion_wxyz=tuple(float(v) for v in quaternion),
                angular_velocity=tuple(float(v) for v in angular),
                linear_acceleration=tuple(float(v) for v in acceleration),
                sequence=after.sequence,
                source_epoch=after.source_epoch,
            ),
            maximum_bracket_gap_ns=maximum_gap,
        )
