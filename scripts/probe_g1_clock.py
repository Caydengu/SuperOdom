#!/usr/bin/env python3
"""Measure the G1-to-Oslo wall-clock offset with bounded SSH round trips."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


def collect_sample(target: str, timeout_sec: float) -> dict[str, int]:
    oslo_send_ns = time.time_ns()
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout_sec:g}", target, "date +%s%N"],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    oslo_receive_ns = time.time_ns()
    robot_realtime_ns = int(completed.stdout.strip())
    oslo_midpoint_ns = (oslo_send_ns + oslo_receive_ns) // 2
    return {
        "oslo_send_realtime_ns": oslo_send_ns,
        "robot_realtime_ns": robot_realtime_ns,
        "oslo_receive_realtime_ns": oslo_receive_ns,
        "round_trip_ns": oslo_receive_ns - oslo_send_ns,
        "robot_minus_oslo_midpoint_ns": robot_realtime_ns - oslo_midpoint_ns,
    }


def summarize(samples: list[dict[str, int]], target: str) -> dict[str, Any]:
    best = min(samples, key=lambda sample: sample["round_trip_ns"])
    return {
        "schema": "g1_oslo_clock_probe_v1",
        "target": target,
        "samples": samples,
        "best_sample_by_minimum_round_trip": best,
        "estimated_robot_minus_oslo_ns": best["robot_minus_oslo_midpoint_ns"],
        "minimum_round_trip_ns": best["round_trip_ns"],
        "alignment_contract": "fit robot/header time to Oslo receipt time; do not assume synchronized clocks",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="SSH target, for example unitree@192.168.123.164")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--timeout-sec", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 3 or args.samples > 31:
        raise SystemExit("--samples must be from 3 through 31")
    if args.output.exists():
        raise FileExistsError(args.output)
    measurements = [collect_sample(args.target, args.timeout_sec) for _ in range(args.samples)]
    result = summarize(measurements, args.target)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
