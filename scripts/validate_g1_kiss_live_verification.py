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


def load_optional_json(path: Path) -> dict[str, Any]:
    """Return an explicit failed artifact instead of aborting validation."""
    try:
        return load_json(path)
    except (OSError, TypeError, ValueError) as error:
        return {
            "status": "missing_or_invalid",
            "artifact": str(path),
            "error": f"{type(error).__name__}: {error}",
        }


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


def last_schema_status(log_path: Path, schema: str) -> dict[str, Any]:
    selected: dict[str, Any] | None = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if f'"schema": "{schema}"' not in line:
            continue
        start = line.find("{")
        if start < 0:
            continue
        candidate = json.loads(line[start:])
        if isinstance(candidate, dict):
            selected = candidate
    if selected is None:
        raise ValueError(f"log contains no {schema} status event")
    return selected


def last_live_status(log_path: Path) -> dict[str, Any]:
    return last_schema_status(log_path, "g1_kiss_live_localization_status_v1")


def process_statuses_ok(
    statuses: dict[str, Any], *, integrated_robot_vlm: bool
) -> bool:
    for name, raw_status in statuses.items():
        status = int(raw_status)
        if name == "live_stack" and integrated_robot_vlm:
            # The integrated launcher deliberately terminates the longer-lived
            # producer after the bounded recorder and robot-vlm observer finish.
            if status in {0, 130, 143}:
                continue
        if status != 0:
            return False
    return True


def validate(run_dir: Path) -> dict[str, Any]:
    manifest = load_json(run_dir / "manifest.json")
    duration_sec = float(manifest["duration_sec"])
    motive_mode = str(manifest.get("motive_mode", "required"))
    if motive_mode not in {"required", "unmapped", "disabled"}:
        raise ValueError(f"unsupported Motive mode: {motive_mode}")
    process_status = load_json(run_dir / "process_status.json")
    motive = (
        load_json(run_dir / "motive" / "summary.json")
        if motive_mode != "disabled"
        else {
            "status": "disabled",
            "role": "not an online localization input",
        }
    )
    lowstate = load_json(run_dir / "lowstate" / "summary.json")
    counts = bag_counts(run_dir / "rosbag" / "live_localization" / "metadata.yaml")
    live = last_live_status(run_dir / "logs" / "live_stack.log")
    stats = live["stats"]
    published = int(stats["published"])
    lidar = int(stats["lidar"])
    availability = published / lidar if lidar else 0.0
    process_ok = process_statuses_ok(
        process_status,
        integrated_robot_vlm=bool(manifest.get("integrated_robot_vlm")),
    )
    motive_ok = motive_mode == "disabled" or (
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
    total_runtime = live.get("stage_runtime_ms", {}).get("total", {})
    pose_age = live.get("pose_age_ms", {})
    timing_ok = (
        float(total_runtime.get("p95", float("inf"))) <= 20.0
        and float(total_runtime.get("p99", float("inf"))) <= 30.0
        and float(pose_age.get("p95", float("inf"))) <= 50.0
    )
    checks = {
        "processes": {"ok": process_ok, "statuses": process_status},
        "motive": {"ok": motive_ok, "mode": motive_mode, **motive},
        "lowstate": {"ok": lowstate_ok, **lowstate},
        "rosbag": {"ok": bag_ok, "message_counts": counts},
        "live_localization": {
            "ok": live_ok,
            "availability": availability,
            "last_status": live,
        },
        "live_timing": {
            "ok": timing_ok,
            "total_runtime_ms": total_runtime,
            "pose_age_ms": pose_age,
            "gates": {
                "maximum_total_runtime_p95_ms": 20.0,
                "maximum_total_runtime_p99_ms": 30.0,
                "maximum_pose_age_p95_ms": 50.0,
            },
        },
    }
    if bool(manifest.get("integrated_robot_vlm")):
        map_readiness = load_json(run_dir / "runtime" / "map-readiness.json")
        map_status = last_schema_status(
            run_dir / "logs" / "map_stack.log",
            "g1_structural_map_localization_status_v1",
        )
        robot_vlm = load_optional_json(run_dir / "robot-vlm-real-backend.json")
        expected_map_sha256 = str(manifest["structural_map_sha256"])
        expected_artifact_sha256 = str(manifest["map_artifact_sha256"])
        map_runtime = map_status.get("attempt_runtime_ms", {})
        map_ready_ok = (
            map_readiness.get("status") == "ready"
            and map_readiness.get("map_digest") == expected_map_sha256
            and int(map_readiness.get("local_source_epoch", -1))
            == int(manifest.get("localization_source_epoch", -2))
            and map_readiness.get("command_capability") == "structurally_unavailable"
        )
        map_lane_ok = (
            bool(map_status.get("initialized"))
            and map_status.get("map_digest") == expected_map_sha256
            and map_status.get("initialization_mode") == "ui"
            and bool(map_status.get("ui_receipt_content_sha256"))
            and int(map_status.get("invalid_root_packets", 1)) == 0
            and int(map_status.get("stats", {}).get("ui_receipts_accepted", 0)) == 1
            and int(map_status.get("stats", {}).get("packets_published", 0)) >= 1
            and float(map_runtime.get("p95", float("inf"))) <= 150.0
            and map_status.get("command_capability") == "structurally_unavailable"
        )
        robot_vlm_ok = (
            robot_vlm.get("status") == "pass"
            and robot_vlm.get("command_capability") == "structurally_unavailable"
            and robot_vlm.get("map", {}).get("artifact_sha256") == expected_artifact_sha256
            and float(robot_vlm.get("base_pose_availability", 0.0)) >= 0.98
            and int(robot_vlm.get("transport", {}).get("local_rejections", 1)) == 0
            and int(robot_vlm.get("transport", {}).get("map_rejections", 1)) == 0
        )
        checks.update(
            {
                "map_readiness": {"ok": map_ready_ok, **map_readiness},
                "structural_map_lane": {
                    "ok": map_lane_ok,
                    "last_status": map_status,
                    "maximum_attempt_runtime_p95_ms": 150.0,
                },
                "robot_vlm_real_backend": {"ok": robot_vlm_ok, **robot_vlm},
            }
        )
    passed = all(bool(check["ok"]) for check in checks.values())
    evidence_class = {
        "stationary": "passive stationary live-localization capture",
        "amo-walk": "operator-controlled AMO live-localization capture",
        "amo-stress": "operator-controlled AMO stress live-localization capture",
    }.get(str(manifest.get("capture_class", "stationary")), "live-localization capture")
    if motive_mode == "disabled":
        evidence_class += " without external ground truth"
    elif motive_mode == "unmapped":
        evidence_class += " with evaluator-only unmapped external ground truth"
    return {
        "schema": "g1_kiss_live_verification_validation_v1",
        "status": "pass" if passed else "fail",
        "run_dir": str(run_dir.resolve()),
        "checks": checks,
        "evidence_class": evidence_class if passed else "incomplete passive capture",
        "absolute_map_pose_accuracy_evaluated": False,
        "external_ground_truth_recorded": motive_mode != "disabled",
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
