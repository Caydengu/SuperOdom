#!/usr/bin/env python3
"""Validate a passive G1 KISS live-localization capture before scoring pose."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

TOPICS = (
    "/utlidar/cloud_livox_mid360",
    "/utlidar/imu_livox_mid360",
    "/g1/localization/pelvis_odom",
    "/g1/localization/cloud_registered",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def bag_counts(metadata_path: Path) -> dict[str, int]:
    text = metadata_path.read_text(encoding="utf-8")
    counts: dict[str, int] = {}
    for topic in TOPICS:
        match = re.search(
            rf"name:\s*{re.escape(topic)}\b[\s\S]*?message_count:\s*(\d+)",
            text,
        )
        counts[topic] = int(match.group(1)) if match else 0
    return counts


def last_live_status(log_path: Path) -> dict[str, Any]:
    selected: dict[str, Any] | None = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if '"schema": "g1_kiss_live_localization_status_v1"' not in line:
            continue
        start = line.find("{")
        if start < 0:
            continue
        candidate = json.loads(line[start:])
        if isinstance(candidate, dict):
            selected = candidate
    if selected is None:
        raise ValueError("live producer log contains no status event")
    return selected


def validate(run_dir: Path) -> dict[str, Any]:
    manifest = load_json(run_dir / "manifest.json")
    duration_sec = float(manifest["duration_sec"])
    process_status = load_json(run_dir / "process_status.json")
    motive = load_json(run_dir / "motive" / "summary.json")
    lowstate = load_json(run_dir / "lowstate" / "summary.json")
    counts = bag_counts(run_dir / "rosbag" / "live_localization" / "metadata.yaml")
    live = last_live_status(run_dir / "logs" / "live_stack.log")
    stats = live["stats"]
    published = int(stats["published"])
    lidar = int(stats["lidar"])
    availability = published / lidar if lidar else 0.0
    process_ok = all(int(value) == 0 for value in process_status.values())
    motive_ok = (
        bool(motive.get("connected"))
        and int(motive.get("resolved_rigid_body_id", -1)) == 42
        and str(motive.get("rigid_body_name")) == "G1_PELVIS_F_4123"
        and float(motive.get("tracking_coverage", 0.0)) >= 0.99
    )
    lowstate_ok = (
        int(lowstate.get("accepted", 0)) >= duration_sec * 200.0
        and int(lowstate.get("invalid", 1)) == 0
        and int(lowstate.get("sequence_gaps", 1)) == 0
        and int(lowstate.get("duplicates_or_reordered", 1)) == 0
        and int(lowstate.get("actuation_topics_created", 1)) == 0
    )
    bag_ok = (
        counts[TOPICS[0]] >= duration_sec * 8.0
        and counts[TOPICS[1]] >= duration_sec * 150.0
        and counts[TOPICS[2]] >= duration_sec * 8.0
        and counts[TOPICS[3]] >= duration_sec * 8.0
    )
    live_ok = (
        bool(live.get("active"))
        and availability >= 0.95
        and int(stats.get("queue_replaced", 1)) == 0
        and int(stats.get("imu_wait_timeouts", 1)) == 0
        and int(stats.get("joint_wait_timeouts", 1)) == 0
        and int(stats.get("transport_rejected", 1)) == 0
        and int(live.get("lowstate_udp_invalid", 1)) == 0
        and live.get("command_capability") == "structurally_unavailable"
    )
    checks = {
        "processes": {"ok": process_ok, "statuses": process_status},
        "motive": {"ok": motive_ok, **motive},
        "lowstate": {"ok": lowstate_ok, **lowstate},
        "rosbag": {"ok": bag_ok, "message_counts": counts},
        "live_localization": {
            "ok": live_ok,
            "availability": availability,
            "last_status": live,
        },
    }
    passed = all(bool(check["ok"]) for check in checks.values())
    return {
        "schema": "g1_kiss_live_verification_validation_v1",
        "status": "pass" if passed else "fail",
        "run_dir": str(run_dir.resolve()),
        "checks": checks,
        "evidence_class": (
            "passive stationary live-localization capture"
            if passed
            else "incomplete passive capture"
        ),
        "hardware_actuation_clearance": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.run_dir)
    output = args.output or args.run_dir / "validation.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
