from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "probe_g1_clock.py"
SPEC = importlib.util.spec_from_file_location("probe_g1_clock", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summary_selects_minimum_round_trip_sample() -> None:
    samples = [
        {"round_trip_ns": 20, "robot_minus_oslo_midpoint_ns": -100},
        {"round_trip_ns": 5, "robot_minus_oslo_midpoint_ns": -90},
        {"round_trip_ns": 12, "robot_minus_oslo_midpoint_ns": -95},
    ]
    result = MODULE.summarize(samples, "unitree@g1")
    assert result["estimated_robot_minus_oslo_ns"] == -90
    assert result["minimum_round_trip_ns"] == 5
