"""Source-time plausibility gate for local G1 pelvis odometry.

This gate does not repair an estimator or compare it with a map.  It prevents
downstream consumers from treating a physically implausible local trajectory
as healthy while SuperOdometry is still reporting a nominal health bit.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class OdometryHealthConfig:
    window_ns: int = 200_000_000
    max_planar_speed_mps: float = 1.5
    max_yaw_rate_radps: float = 2.0
    recovery_hold_ns: int = 1_000_000_000

    def __post_init__(self) -> None:
        if not isinstance(self.window_ns, int) or self.window_ns <= 0:
            raise ValueError("window_ns must be a positive integer")
        if not isinstance(self.recovery_hold_ns, int) or self.recovery_hold_ns < 0:
            raise ValueError("recovery_hold_ns must be a non-negative integer")
        for name, value in (
            ("max_planar_speed_mps", self.max_planar_speed_mps),
            ("max_yaw_rate_radps", self.max_yaw_rate_radps),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class OdometryHealthState:
    healthy: bool
    reason: str
    source_time_ns: int
    window_duration_ns: int | None
    planar_speed_mps: float | None
    yaw_rate_radps: float | None


@dataclass(frozen=True)
class _PoseSample:
    source_time_ns: int
    x_m: float
    y_m: float
    yaw_rad: float


def _finite_tuple(values: Iterable[float], expected: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != expected or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {expected} finite values")
    return result


def _yaw_from_wxyz(quaternion_wxyz: Iterable[float]) -> float:
    w, x, y, z = _finite_tuple(quaternion_wxyz, 4, "quaternion_wxyz")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
        raise ValueError("quaternion_wxyz must be normalized")
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _shortest_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


class OdometryHealthGate:
    """Reject sustained source-time motion outside a configured G1 envelope."""

    def __init__(self, config: OdometryHealthConfig = OdometryHealthConfig()):
        self.config = config
        self._samples: deque[_PoseSample] = deque()
        self._last_source_time_ns: int | None = None
        self._unhealthy_until_ns = 0

    def reset(self) -> None:
        self._samples.clear()
        self._last_source_time_ns = None
        self._unhealthy_until_ns = 0

    def update(
        self,
        *,
        source_time_ns: int,
        position_xyz_m: Iterable[float],
        quaternion_wxyz: Iterable[float],
    ) -> OdometryHealthState:
        if not isinstance(source_time_ns, int) or source_time_ns < 0:
            raise ValueError("source_time_ns must be a non-negative integer")
        x_m, y_m, _ = _finite_tuple(position_xyz_m, 3, "position_xyz_m")
        yaw_rad = _yaw_from_wxyz(quaternion_wxyz)
        sample = _PoseSample(source_time_ns, x_m, y_m, yaw_rad)

        if (
            self._last_source_time_ns is not None
            and source_time_ns <= self._last_source_time_ns
        ):
            self._samples.clear()
            self._samples.append(sample)
            self._last_source_time_ns = source_time_ns
            self._unhealthy_until_ns = source_time_ns + self.config.recovery_hold_ns
            return OdometryHealthState(
                healthy=False,
                reason="non_monotonic_source_time",
                source_time_ns=source_time_ns,
                window_duration_ns=None,
                planar_speed_mps=None,
                yaw_rate_radps=None,
            )

        self._last_source_time_ns = source_time_ns
        self._samples.append(sample)
        target_time_ns = source_time_ns - self.config.window_ns
        while len(self._samples) >= 2 and self._samples[1].source_time_ns <= target_time_ns:
            self._samples.popleft()

        reference = self._samples[0]
        duration_ns = source_time_ns - reference.source_time_ns
        if duration_ns < self.config.window_ns:
            return OdometryHealthState(
                healthy=False,
                reason="warming_up",
                source_time_ns=source_time_ns,
                window_duration_ns=duration_ns,
                planar_speed_mps=None,
                yaw_rate_radps=None,
            )

        duration_s = duration_ns * 1e-9
        planar_speed_mps = math.hypot(x_m - reference.x_m, y_m - reference.y_m) / duration_s
        yaw_rate_radps = abs(_shortest_angle(yaw_rad - reference.yaw_rad)) / duration_s
        reasons: list[str] = []
        if planar_speed_mps > self.config.max_planar_speed_mps:
            reasons.append("planar_speed")
        if yaw_rate_radps > self.config.max_yaw_rate_radps:
            reasons.append("yaw_rate")
        if reasons:
            self._unhealthy_until_ns = source_time_ns + self.config.recovery_hold_ns
            return OdometryHealthState(
                healthy=False,
                reason="+".join(reasons),
                source_time_ns=source_time_ns,
                window_duration_ns=duration_ns,
                planar_speed_mps=planar_speed_mps,
                yaw_rate_radps=yaw_rate_radps,
            )
        if source_time_ns < self._unhealthy_until_ns:
            return OdometryHealthState(
                healthy=False,
                reason="recovery_hold",
                source_time_ns=source_time_ns,
                window_duration_ns=duration_ns,
                planar_speed_mps=planar_speed_mps,
                yaw_rate_radps=yaw_rate_radps,
            )
        return OdometryHealthState(
            healthy=True,
            reason="plausible",
            source_time_ns=source_time_ns,
            window_duration_ns=duration_ns,
            planar_speed_mps=planar_speed_mps,
            yaw_rate_radps=yaw_rate_radps,
        )
