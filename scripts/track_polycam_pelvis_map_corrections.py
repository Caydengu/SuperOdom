#!/usr/bin/env python3
"""Track fail-closed Polycam map corrections over pelvis-local odometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from g1_root_state_bridge.amo_dataset import load_lowstate
from g1_root_state_bridge.amo_map_localization import (
    load_scan_archive,
    pelvis_structural_cloud,
    refine_icp,
)


def _shortest_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _load_treatment(path: Path, treatment: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            if payload.get("treatment") == treatment:
                records.append(payload)
    if len(records) < 2:
        raise ValueError(f"missing treatment {treatment!r}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--map-key", default="map_xy_all_5cm")
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--query-scans", type=Path, required=True)
    parser.add_argument("--query-treatments", type=Path, required=True)
    parser.add_argument("--query-treatment", required=True)
    parser.add_argument("--lowstate", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--period-sec", type=float, default=2.0)
    parser.add_argument(
        "--correction-mode",
        choices=("se2", "yaw_only_preserve_current_position"),
        default="yaw_only_preserve_current_position",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    with np.load(args.map, allow_pickle=False) as map_archive:
        map_xy = np.asarray(map_archive[args.map_key], dtype=np.float64)
        map_metadata = json.loads(str(map_archive["metadata_json"]))
    initialization = json.loads(args.initialization.read_text(encoding="utf-8"))
    if not initialization.get("healthy"):
        raise ValueError("initialization did not pass its confidence gates")
    scans = load_scan_archive(args.query_scans)
    records = _load_treatment(args.query_treatments, args.query_treatment)
    lowstate = load_lowstate(args.lowstate)
    pose_time = np.asarray([row["source_time_ns"] for row in records], dtype=np.int64)
    pose_xy = np.asarray([row["position_xyz_m"][:2] for row in records], dtype=np.float64)
    rotation = np.asarray(initialization["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(initialization["translation_m"], dtype=np.float64)
    first_available_ns = int(initialization["query_audit"]["end_source_time_ns"])
    corrections: list[dict[str, object]] = [{
        "available_after_source_ns": first_available_ns,
        "accepted": True,
        "kind": "global_initialization",
        "rotation_matrix": rotation.tolist(),
        "translation_m": translation.tolist(),
        "yaw_deg": math.degrees(math.atan2(rotation[0, 1], rotation[0, 0])),
        "quality": {
            "inlier_fraction": initialization["icp"]["inlier_fraction"],
            "rmse_m": initialization["icp"]["rmse_m"],
            "p95_m": initialization["icp"]["p95_m"],
            "observability": initialization["icp"]["observability"],
        },
    }]
    window_ns = round(args.window_sec * 1e9)
    period_ns = round(args.period_sec * 1e9)
    for end_ns in range(
        first_available_ns + period_ns, int(scans.source_time_ns[-1]), period_ns
    ):
        try:
            query_xy, audit = pelvis_structural_cloud(
                scans,
                records,
                lowstate,
                start_source_time_ns=end_ns - window_ns,
                end_source_time_ns=end_ns,
                scan_stride=2,
            )
            candidate = refine_icp(
                map_xy,
                query_xy,
                rotation,
                translation,
                maximum_correspondence_m=0.25,
            )
            candidate_rotation = np.asarray(candidate["rotation"], dtype=np.float64)
            candidate_translation = np.asarray(candidate["translation_m"], dtype=np.float64)
            pose_index = int(np.argmin(np.abs(pose_time - end_ns)))
            old_position = pose_xy[pose_index] @ rotation + translation
            new_position = pose_xy[pose_index] @ candidate_rotation + candidate_translation
            jump_m = float(np.linalg.norm(new_position - old_position))
            old_yaw = math.degrees(math.atan2(rotation[0, 1], rotation[0, 0]))
            new_yaw = math.degrees(
                math.atan2(candidate_rotation[0, 1], candidate_rotation[0, 0])
            )
            yaw_jump_deg = abs(_shortest_degrees(new_yaw - old_yaw))
            accepted = (
                float(candidate["inlier_fraction"]) >= 0.45
                and float(candidate["rmse_m"]) <= 0.12
                and float(candidate["p95_m"]) <= 0.22
                and bool(candidate["observability"]["observable"])
                and jump_m <= 0.25
                and yaw_jump_deg <= 5.0
            )
            applied_rotation = candidate_rotation
            applied_translation = candidate_translation
            applied_jump_m = jump_m
            if args.correction_mode == "yaw_only_preserve_current_position":
                applied_translation = old_position - pose_xy[pose_index] @ applied_rotation
                applied_jump_m = float(
                    np.linalg.norm(
                        pose_xy[pose_index] @ applied_rotation
                        + applied_translation
                        - old_position
                    )
                )
            correction = {
                "available_after_source_ns": end_ns,
                "accepted": accepted,
                "kind": "local_icp_correction",
                "rotation_matrix": applied_rotation.tolist(),
                "translation_m": applied_translation.tolist(),
                "yaw_deg": new_yaw,
                "quality": {
                    "inlier_fraction": float(candidate["inlier_fraction"]),
                    "rmse_m": float(candidate["rmse_m"]),
                    "p95_m": float(candidate["p95_m"]),
                    "pose_jump_m": jump_m,
                    "applied_pose_jump_m": applied_jump_m,
                    "yaw_jump_deg": yaw_jump_deg,
                    "candidate_translation_m": candidate_translation.tolist(),
                    "query_point_count": int(query_xy.shape[0]),
                    "observability": candidate["observability"],
                    "query_audit": audit,
                },
            }
        except ValueError as error:
            correction = {
                "available_after_source_ns": end_ns,
                "accepted": False,
                "kind": "local_icp_correction",
                "rejection_reason": str(error),
                "rotation_matrix": rotation.tolist(),
                "translation_m": translation.tolist(),
                "yaw_deg": math.degrees(math.atan2(rotation[0, 1], rotation[0, 0])),
                "quality": {},
            }
        corrections.append(correction)
        if correction["accepted"]:
            rotation = np.asarray(correction["rotation_matrix"], dtype=np.float64)
            translation = np.asarray(correction["translation_m"], dtype=np.float64)
    report = {
        "schema": "g1_polycam_confidence_gated_map_corrections_v1",
        "causal": True,
        "motive_online_input": False,
        "map_id": map_metadata["map_id"],
        "map_key": args.map_key,
        "map_role": "candidate_polycam_prior_not_ground_truth",
        "query_treatment": args.query_treatment,
        "window_sec": args.window_sec,
        "period_sec": args.period_sec,
        "correction_mode": args.correction_mode,
        "gates": {
            "maximum_correspondence_m": 0.25,
            "minimum_inlier_fraction": 0.45,
            "maximum_rmse_m": 0.12,
            "maximum_p95_m": 0.22,
            "minimum_observability_eigenvalue": 0.02,
            "maximum_observability_condition_number": 1e6,
            "maximum_pose_jump_m": 0.25,
            "maximum_yaw_jump_deg": 5.0,
        },
        "attempt_count": len(corrections) - 1,
        "accepted_count": sum(bool(row["accepted"]) for row in corrections[1:]),
        "rejected_count": sum(not bool(row["accepted"]) for row in corrections[1:]),
        "corrections": corrections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "attempt_count": report["attempt_count"],
        "accepted_count": report["accepted_count"],
        "rejected_count": report["rejected_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
