#!/usr/bin/env python3
"""Create a provenance-marked replay archive with one controlled IMU gap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-gap-ms", type=float, default=30.0)
    parser.add_argument(
        "--selection",
        choices=("maximum-angular-speed", "middle"),
        default="maximum-angular-speed",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if not 25.0 < args.target_gap_ms <= 40.0:
        raise ValueError("target gap must exercise the configured (25, 40] ms bridge")

    archive = np.load(args.input, allow_pickle=False)
    arrays = {name: np.asarray(archive[name]) for name in archive.files}
    source_time = np.asarray(arrays["imu_source_time_ns"], dtype=np.int64)
    receipt_time = np.asarray(arrays["imu_receipt_time_ns"], dtype=np.int64)
    angular = np.asarray(arrays["imu_angular_velocity_radps"], dtype=np.float64)
    if source_time.size < 10 or np.any(np.diff(source_time) <= 0):
        raise ValueError("IMU source timestamps must be sufficiently long and increasing")

    if args.selection == "maximum-angular-speed":
        # Exclude capture boundaries and select the strongest rotational motion.
        margin = max(10, source_time.size // 20)
        center = margin + int(np.argmax(np.linalg.norm(angular[margin:-margin], axis=1)))
    else:
        center = source_time.size // 2
    target_ns = round(args.target_gap_ms * 1e6)
    left = center
    right = center + 1
    while right < source_time.size - 1 and source_time[right] - source_time[left] < target_ns:
        right += 1
    actual_gap_ns = int(source_time[right] - source_time[left])
    if not 25_000_000 < actual_gap_ns <= 40_000_000:
        raise ValueError(f"could not construct an admitted bridge gap; got {actual_gap_ns} ns")

    # Retain both endpoints so the bridge uses only observations that genuinely
    # bracket the missing interval.  Remove interior samples from every IMU array.
    keep = np.ones(source_time.size, dtype=bool)
    keep[left + 1 : right] = False
    imu_keys = [name for name, value in arrays.items() if name.startswith("imu_") and value.shape[:1] == source_time.shape]
    for name in imu_keys:
        arrays[name] = arrays[name][keep]

    metadata = {}
    if "metadata_json" in arrays:
        metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["controlled_imu_gap"] = {
        "schema": "g1_controlled_replay_imu_gap_v1",
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "selection": args.selection,
        "left_source_time_ns": int(source_time[left]),
        "right_source_time_ns": int(source_time[right]),
        "actual_gap_ns": actual_gap_ns,
        "removed_samples": int(right - left - 1),
        "angular_speed_radps_at_left": float(np.linalg.norm(angular[left])),
        "angular_speed_radps_at_right": float(np.linalg.norm(angular[right])),
        "receipt_gap_ns": int(receipt_time[right] - receipt_time[left]),
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(json.dumps({"output": str(args.output), **metadata["controlled_imu_gap"]}, sort_keys=True))


if __name__ == "__main__":
    main()
