#!/usr/bin/env python3
"""Validate the minimum offline-localization dataset contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def check_dataset(run_dir: Path) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    motive_path = run_dir / "motive" / "summary.json"
    if motive_path.is_file():
        motive = load_json(motive_path)
        motive_ok = bool(motive.get("connected")) and int(motive.get("frames_written", 0)) > 0
        motive_ok = motive_ok and float(motive.get("tracking_coverage", 0.0)) >= 0.95
        checks["motive"] = {"ok": motive_ok, **motive}
    else:
        checks["motive"] = {"ok": False, "reason": "missing summary.json"}

    lowstate_path = run_dir / "lowstate" / "summary.json"
    if lowstate_path.is_file():
        lowstate = load_json(lowstate_path)
        lowstate_ok = int(lowstate.get("accepted", 0)) > 0
        lowstate_ok = lowstate_ok and int(lowstate.get("invalid", 1)) == 0
        lowstate_ok = lowstate_ok and int(lowstate.get("sequence_gaps", 1)) == 0
        lowstate_ok = lowstate_ok and float(lowstate.get("accepted_rate_hz", 0.0)) >= 200.0
        checks["lowstate_and_pelvis_imu"] = {"ok": lowstate_ok, **lowstate}
    else:
        checks["lowstate_and_pelvis_imu"] = {"ok": False, "reason": "missing summary.json"}

    lidar_metadata = run_dir / "lidar" / "data" / "live_input_probe" / "metadata.yaml"
    if lidar_metadata.is_file():
        text = lidar_metadata.read_text(encoding="utf-8")
        lidar_counts: dict[str, int] = {}
        for topic in ("/livox/lidar", "/livox/imu"):
            pattern = rf"name:\s*{re.escape(topic)}\b[\s\S]*?message_count:\s*(\d+)"
            match = re.search(pattern, text)
            lidar_counts[topic] = int(match.group(1)) if match else 0
        lidar_ok = lidar_counts["/livox/lidar"] > 0 and lidar_counts["/livox/imu"] > 0
        checks["lidar_and_livox_imu"] = {"ok": lidar_ok, "message_counts": lidar_counts}
    else:
        checks["lidar_and_livox_imu"] = {"ok": False, "reason": "missing metadata.yaml"}

    vision_dir = run_dir / "vision" / "capture"
    required_vision = ("recording.mp4", "timestamps.npz", "depth.npz", "depth.mp4")
    vision_sizes = {
        name: (vision_dir / name).stat().st_size if (vision_dir / name).is_file() else 0
        for name in required_vision
    }
    checks["rgbd"] = {"ok": all(size > 0 for size in vision_sizes.values()), "sizes_bytes": vision_sizes}

    process_status_path = run_dir / "process_status.json"
    if process_status_path.is_file():
        process_status = load_json(process_status_path)
        checks["processes"] = {
            "ok": all(int(value) == 0 for value in process_status.values()),
            "statuses": process_status,
        }
    else:
        checks["processes"] = {"ok": False, "reason": "missing process_status.json"}

    passed = all(bool(check["ok"]) for check in checks.values())
    return {
        "schema": "g1_motive_localization_dataset_validation_v1",
        "run_dir": str(run_dir.resolve()),
        "status": "pass" if passed else "fail",
        "checks": checks,
        "evidence_class": "passive stationary capture" if passed else "incomplete capture",
        "hardware_actuation_clearance": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check_dataset(args.run_dir)
    output = args.output or args.run_dir / "validation.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
