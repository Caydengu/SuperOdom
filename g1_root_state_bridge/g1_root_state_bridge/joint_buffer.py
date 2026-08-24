"""Deterministic synchronization buffer for timestamped G1 joints."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from g1_root_state_bridge.joint_contract import TimedJointSample


class JointSynchronizationError(ValueError):
    """No trustworthy joint sample can be produced for a query time."""


@dataclass(frozen=True)
class JointSynchronization:
    """Interpolated joints plus the real samples that bounded the query."""

    sample: TimedJointSample
    before_time_ns: int
    after_time_ns: int
    representative_time_ns: int
    representative_sequence: int

    @property
    def before_gap_ns(self) -> int:
        return self.sample.stamp_ns - self.before_time_ns

    @property
    def after_gap_ns(self) -> int:
        return self.after_time_ns - self.sample.stamp_ns

    @property
    def signed_gap_ns(self) -> int:
        return self.representative_time_ns - self.sample.stamp_ns

    @property
    def maximum_bracket_gap_ns(self) -> int:
        return max(self.before_gap_ns, self.after_gap_ns)


class JointBuffer:
    def __init__(self, capacity: int = 1024):
        if capacity < 2:
            raise ValueError("JointBuffer capacity must be at least two")
        self._samples: deque[TimedJointSample] = deque(maxlen=capacity)

    def append(self, sample: TimedJointSample) -> None:
        """Validate, canonicalize, and append a strictly monotonic sample."""
        sample = sample.canonicalized()
        if self._samples:
            latest = self._samples[-1]
            if sample.source_epoch < latest.source_epoch:
                raise JointSynchronizationError("joint source epoch is not increasing")
            if sample.source_epoch > latest.source_epoch:
                self._samples.clear()
                self._samples.append(sample)
                return
            if sample.stamp_ns <= latest.stamp_ns:
                raise JointSynchronizationError("joint timestamp is not strictly increasing")
            if sample.sequence <= latest.sequence:
                raise JointSynchronizationError("joint sequence is not strictly increasing")
        self._samples.append(sample)

    def clear(self) -> None:
        """Discard all samples after an estimator/source reset."""
        self._samples.clear()

    @property
    def time_bounds_ns(self) -> tuple[int, int] | None:
        """Return the admitted source-time interval without exposing samples."""
        if not self._samples:
            return None
        return self._samples[0].stamp_ns, self._samples[-1].stamp_ns

    def interpolate(self, query_ns: int, max_gap_ns: int) -> TimedJointSample:
        """Linearly interpolate a bracketed sample within the declared gap."""
        return self.synchronize(query_ns, max_gap_ns).sample

    def synchronize(self, query_ns: int, max_gap_ns: int) -> JointSynchronization:
        """Interpolate while retaining the observable source-time separation."""
        if max_gap_ns < 0:
            raise ValueError("max_gap_ns must be non-negative")
        if not self._samples:
            raise JointSynchronizationError("joint buffer is empty")
        if query_ns < self._samples[0].stamp_ns or query_ns > self._samples[-1].stamp_ns:
            raise JointSynchronizationError("joint query is outside the buffered interval")

        before: TimedJointSample | None = None
        after: TimedJointSample | None = None
        for sample in self._samples:
            if sample.stamp_ns == query_ns:
                return JointSynchronization(
                    sample=sample,
                    before_time_ns=query_ns,
                    after_time_ns=query_ns,
                    representative_time_ns=query_ns,
                    representative_sequence=sample.sequence,
                )
            if sample.stamp_ns < query_ns:
                before = sample
                continue
            after = sample
            break

        if before is None or after is None:
            raise JointSynchronizationError("joint query is outside the buffered interval")
        before_gap = query_ns - before.stamp_ns
        after_gap = after.stamp_ns - query_ns
        if before_gap > max_gap_ns or after_gap > max_gap_ns:
            raise JointSynchronizationError(
                f"joint interpolation gap exceeds {max_gap_ns} ns "
                f"(before={before_gap}, after={after_gap})"
            )

        alpha = before_gap / (after.stamp_ns - before.stamp_ns)
        position = tuple(
            left + alpha * (right - left)
            for left, right in zip(before.position, after.position, strict=True)
        )
        velocity = tuple(
            left + alpha * (right - left)
            for left, right in zip(before.velocity, after.velocity, strict=True)
        )
        synchronized_sample = TimedJointSample(
            stamp_ns=query_ns,
            receipt_ns=max(before.receipt_ns, after.receipt_ns),
            names=before.names,
            position=position,
            velocity=velocity,
            sequence=after.sequence,
            source_epoch=after.source_epoch,
        )
        representative_time_ns = (
            before.stamp_ns if before_gap <= after_gap else after.stamp_ns
        )
        representative_sequence = (
            before.sequence if before_gap <= after_gap else after.sequence
        )
        return JointSynchronization(
            sample=synchronized_sample,
            before_time_ns=before.stamp_ns,
            after_time_ns=after.stamp_ns,
            representative_time_ns=representative_time_ns,
            representative_sequence=representative_sequence,
        )
