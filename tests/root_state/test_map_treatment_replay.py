from __future__ import annotations

import numpy as np

from scripts.replay_structural_map_corrections import _window_is_stationary


def _row(time_ns: int, speed: float, yaw_rate: float) -> dict[str, object]:
    return {
        "source_time_ns": time_ns,
        "linear_velocity_xyz_mps": [speed, 0.0, 0.0],
        "angular_velocity_xyz_radps": [0.0, 0.0, yaw_rate],
    }


def test_stationary_window_requires_full_fresh_coverage() -> None:
    rows = [_row(index * 100_000_000, 0.01, 0.02) for index in range(41)]
    times = np.asarray([row["source_time_ns"] for row in rows], dtype=np.int64)
    stationary, audit = _window_is_stationary(
        rows,
        times,
        end_time_ns=4_000_000_000,
        window_sec=5.0,
        maximum_linear_speed_mps=0.05,
        maximum_yaw_rate_radps=0.15,
    )
    assert not stationary
    assert audit["coverage_sec"] == 4.0


def test_stationary_window_uses_robust_onboard_velocity_gate() -> None:
    rows = [_row(index * 100_000_000, 0.01, 0.02) for index in range(51)]
    rows[10] = _row(1_000_000_000, 0.20, 0.50)
    times = np.asarray([row["source_time_ns"] for row in rows], dtype=np.int64)
    stationary, audit = _window_is_stationary(
        rows,
        times,
        end_time_ns=5_000_000_000,
        window_sec=5.0,
        maximum_linear_speed_mps=0.05,
        maximum_yaw_rate_radps=0.15,
    )
    assert stationary
    assert audit["linear_speed_p90_mps"] == 0.01
    assert audit["yaw_rate_p90_radps"] == 0.02


def test_stationary_window_rejects_sustained_motion() -> None:
    rows = [_row(index * 100_000_000, 0.15, 0.40) for index in range(51)]
    times = np.asarray([row["source_time_ns"] for row in rows], dtype=np.int64)
    stationary, audit = _window_is_stationary(
        rows,
        times,
        end_time_ns=5_000_000_000,
        window_sec=5.0,
        maximum_linear_speed_mps=0.05,
        maximum_yaw_rate_radps=0.15,
    )
    assert not stationary
    assert audit["linear_speed_p90_mps"] == 0.15
