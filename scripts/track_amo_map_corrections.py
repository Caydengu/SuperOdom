#!/usr/bin/env python3
"""Track confidence-gated slow map corrections after global initialization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from g1_root_state_bridge.amo_map_localization import (
    load_scan_archive,
    refine_icp,
    structural_cloud,
)
from g1_root_state_bridge.amo_treatments import load_odometry_track


def shortest_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-clouds", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--query-scans", type=Path, required=True)
    parser.add_argument("--query-odometry", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--period-sec", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    with np.load(args.map_clouds, allow_pickle=False) as clouds:
        map_xy = np.asarray(clouds["map_xy"], dtype=np.float64)
    initialization = json.loads(args.initialization.read_text(encoding="utf-8"))
    if not initialization.get("healthy"):
        raise ValueError("initialization did not pass its confidence gates")
    archive = load_scan_archive(args.query_scans)
    odometry = load_odometry_track(args.query_odometry).records
    odom_time = np.asarray([row["source_time_ns"] for row in odometry], dtype=np.int64)
    odom_xy = np.asarray(
        [row["position_xyz_m"][:2] for row in odometry], dtype=np.float64
    )
    rotation = np.asarray(initialization["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(initialization["translation_m"], dtype=np.float64)
    first_available = int(initialization["query_audit"]["end_source_time_ns"])
    corrections = [
        {
            "available_after_source_ns": first_available,
            "accepted": True,
            "kind": "global_initialization",
            "rotation_matrix": rotation.tolist(),
            "translation_m": translation.tolist(),
            "yaw_deg": float(math.degrees(math.atan2(rotation[0, 1], rotation[0, 0]))),
            "quality": {
                "inlier_fraction": initialization["icp"]["inlier_fraction"],
                "p95_m": initialization["icp"]["p95_m"],
            },
        }
    ]
    period_ns = round(args.period_sec * 1e9)
    window_ns = round(args.window_sec * 1e9)
    for end_ns in range(
        first_available + period_ns, int(archive.source_time_ns[-1]), period_ns
    ):
        query_xy, audit = structural_cloud(
            archive,
            odometry,
            start_source_time_ns=end_ns - window_ns,
            end_source_time_ns=end_ns,
            scan_stride=2,
        )
        candidate = refine_icp(map_xy, query_xy, rotation, translation)
        candidate_rotation = np.asarray(candidate["rotation"], dtype=np.float64)
        candidate_translation = np.asarray(candidate["translation_m"], dtype=np.float64)
        odom_index = int(np.argmin(np.abs(odom_time - end_ns)))
        old_position = odom_xy[odom_index] @ rotation + translation
        new_position = odom_xy[odom_index] @ candidate_rotation + candidate_translation
        jump_m = float(np.linalg.norm(new_position - old_position))
        old_yaw = math.degrees(math.atan2(rotation[0, 1], rotation[0, 0]))
        new_yaw = math.degrees(
            math.atan2(candidate_rotation[0, 1], candidate_rotation[0, 0])
        )
        yaw_jump_deg = abs(shortest_degrees(new_yaw - old_yaw))
        accepted = (
            float(candidate["inlier_fraction"]) >= 0.50
            and float(candidate["p95_m"]) <= 0.15
            and jump_m <= 0.25
            and yaw_jump_deg <= 3.0
        )
        correction = {
            "available_after_source_ns": end_ns,
            "accepted": accepted,
            "kind": "local_icp_correction",
            "rotation_matrix": candidate_rotation.tolist(),
            "translation_m": candidate_translation.tolist(),
            "yaw_deg": new_yaw,
            "quality": {
                "inlier_fraction": float(candidate["inlier_fraction"]),
                "rmse_m": float(candidate["rmse_m"]),
                "p95_m": float(candidate["p95_m"]),
                "pose_jump_m": jump_m,
                "yaw_jump_deg": yaw_jump_deg,
                "query_point_count": int(query_xy.shape[0]),
                "pose_match_age_ms_p95": audit["pose_match_age_ms_p95"],
            },
        }
        corrections.append(correction)
        if accepted:
            rotation = candidate_rotation
            translation = candidate_translation
    report = {
        "schema": "g1_confidence_gated_map_corrections_v1",
        "causal": True,
        "motive_online_input": False,
        "window_sec": args.window_sec,
        "period_sec": args.period_sec,
        "gates": {
            "minimum_inlier_fraction": 0.50,
            "maximum_p95_m": 0.15,
            "maximum_pose_jump_m": 0.25,
            "maximum_yaw_jump_deg": 3.0,
        },
        "attempt_count": len(corrections) - 1,
        "accepted_count": sum(bool(row["accepted"]) for row in corrections[1:]),
        "rejected_count": sum(not bool(row["accepted"]) for row in corrections[1:]),
        "corrections": corrections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("attempt_count", "accepted_count", "rejected_count")
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
