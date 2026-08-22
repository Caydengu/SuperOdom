from __future__ import annotations

import math

import pytest

from g1_root_state_bridge.odometry_health import (
    OdometryHealthConfig,
    OdometryHealthGate,
)


def _quaternion(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _update(
    gate: OdometryHealthGate,
    time_ns: int,
    x_m: float,
    yaw_rad: float = 0.0,
):
    return gate.update(
        source_time_ns=time_ns,
        position_xyz_m=(x_m, 0.0, 0.8),
        quaternion_wxyz=_quaternion(yaw_rad),
    )


def test_gate_is_fail_closed_during_source_time_warmup() -> None:
    gate = OdometryHealthGate(OdometryHealthConfig(window_ns=200_000_000))

    first = _update(gate, 1_000_000_000, 0.0)
    second = _update(gate, 1_100_000_000, 0.02)
    ready = _update(gate, 1_200_000_000, 0.04)

    assert not first.healthy and first.reason == "warming_up"
    assert not second.healthy and second.reason == "warming_up"
    assert ready.healthy and ready.reason == "plausible"
    assert ready.planar_speed_mps == pytest.approx(0.2)


def test_gate_rejects_implausible_translation_and_requires_recovery() -> None:
    gate = OdometryHealthGate(
        OdometryHealthConfig(
            window_ns=200_000_000,
            max_planar_speed_mps=1.5,
            recovery_hold_ns=300_000_000,
        )
    )
    _update(gate, 1_000_000_000, 0.0)
    rejected = _update(gate, 1_200_000_000, 0.4)
    held = _update(gate, 1_400_000_000, 0.41)
    recovered = _update(gate, 1_501_000_000, 0.42)

    assert not rejected.healthy and rejected.reason == "planar_speed"
    assert rejected.planar_speed_mps == pytest.approx(2.0)
    assert not held.healthy and held.reason == "recovery_hold"
    assert recovered.healthy


def test_gate_uses_shortest_yaw_arc_and_rejects_large_yaw_rate() -> None:
    gate = OdometryHealthGate(
        OdometryHealthConfig(window_ns=200_000_000, max_yaw_rate_radps=2.0)
    )
    _update(gate, 1_000_000_000, 0.0, math.radians(179.0))
    wrapped = _update(gate, 1_200_000_000, 0.0, math.radians(-179.0))
    rejected = _update(gate, 1_400_000_000, 0.0, math.radians(-120.0))

    assert wrapped.healthy
    assert wrapped.yaw_rate_radps == pytest.approx(math.radians(10.0))
    assert not rejected.healthy and rejected.reason == "yaw_rate"


def test_gate_fails_closed_and_rewarms_on_non_monotonic_time() -> None:
    gate = OdometryHealthGate(OdometryHealthConfig(window_ns=100_000_000))
    _update(gate, 1_000_000_000, 0.0)
    assert _update(gate, 1_100_000_000, 0.01).healthy

    reset = _update(gate, 1_050_000_000, 0.02)
    warming = _update(gate, 1_100_000_000, 0.02)

    assert not reset.healthy and reset.reason == "non_monotonic_source_time"
    assert not warming.healthy and warming.reason == "warming_up"


@pytest.mark.parametrize(
    "config",
    [
        OdometryHealthConfig(window_ns=1),
        OdometryHealthConfig(recovery_hold_ns=0),
    ],
)
def test_valid_configurations(config: OdometryHealthConfig) -> None:
    assert OdometryHealthGate(config)


def test_invalid_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="window_ns"):
        OdometryHealthConfig(window_ns=0)
    with pytest.raises(ValueError, match="max_planar_speed_mps"):
        OdometryHealthConfig(max_planar_speed_mps=math.inf)
