#!/usr/bin/env python3
"""Apply fixed map x/y initialization plus independently corrected heading."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from g1_root_state_bridge.amo_treatments import rotation_from_xyzw, xyzw_from_rotation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatments", type=Path, required=True)
    parser.add_argument("--source-treatment", required=True)
    parser.add_argument("--output-treatment", required=True)
    parser.add_argument("--map-localization", type=Path, required=True)
    parser.add_argument("--map-heading-corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    initialization = json.loads(args.map_localization.read_text(encoding="utf-8"))
    correction_report = json.loads(args.map_heading_corrections.read_text(encoding="utf-8"))
    if not initialization.get("healthy"):
        raise ValueError("map initialization is not healthy")
    if (
        not correction_report.get("causal")
        or correction_report.get("motive_online_input")
        or correction_report.get("correction_mode")
        != "yaw_only_preserve_current_position"
    ):
        raise ValueError("heading corrections violate the causal yaw-only contract")
    initial_rotation = np.asarray(initialization["rotation_matrix"], dtype=np.float64)
    initial_translation = np.asarray(initialization["translation_m"], dtype=np.float64)
    available_after_ns = int(initialization["query_audit"]["end_source_time_ns"])
    corrections = [row for row in correction_report["corrections"] if row["accepted"]]
    if not corrections:
        raise ValueError("correction report contains no accepted transforms")
    output_count = 0
    with (
        args.treatments.open(encoding="utf-8") as source,
        args.output.open("x", encoding="utf-8") as output,
    ):
        for line in source:
            payload = json.loads(line)
            if payload.get("kind") == "metadata":
                payload["decoupled_map_pose"] = {
                    "schema": "g1_decoupled_polycam_pose_v1",
                    "motive_online_input": False,
                    "position": "fixed_initial_map_T_local_applied_to_local_xy",
                    "heading": "latest_accepted_observable_map_icp_yaw_plus_local_heading",
                    "map_localization": str(args.map_localization),
                    "map_heading_corrections": str(args.map_heading_corrections),
                    "available_after_source_ns": available_after_ns,
                }
            output.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            if payload.get("treatment") != args.source_treatment:
                continue
            source_time_ns = int(payload["source_time_ns"])
            if source_time_ns < available_after_ns:
                continue
            active_index = max(
                index
                for index, row in enumerate(corrections)
                if int(row["available_after_source_ns"]) <= source_time_ns
            )
            active = corrections[active_index]
            heading_rotation_xy = np.asarray(active["rotation_matrix"], dtype=np.float64)
            heading_angle = math.atan2(
                float(heading_rotation_xy[0, 1]), float(heading_rotation_xy[0, 0])
            )
            c, s = math.cos(heading_angle), math.sin(heading_angle)
            heading_rotation = np.asarray(
                ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)),
                dtype=np.float64,
            )
            derived = dict(payload)
            derived["treatment"] = args.output_treatment
            derived["frame_id"] = "polycam_map"
            derived["child_frame_id"] = "pelvis"
            position = np.asarray(payload["position_xyz_m"], dtype=np.float64)
            position[:2] = position[:2] @ initial_rotation + initial_translation
            derived["position_xyz_m"] = position.tolist()
            derived["quaternion_xyzw"] = list(
                xyzw_from_rotation(
                    heading_rotation
                    @ rotation_from_xyzw(np.asarray(payload["quaternion_xyzw"]))
                )
            )
            derived["map_initialized"] = True
            derived["map_correction_index"] = active_index
            derived["position_correction_mode"] = "fixed_global_initialization"
            derived["heading_correction_mode"] = "latest_accepted_map_icp_yaw"
            output.write(json.dumps(derived, sort_keys=True, separators=(",", ":")) + "\n")
            output_count += 1
    if output_count == 0:
        raise ValueError("no source records occur after map initialization")
    print(json.dumps({
        "output": str(args.output),
        "output_count": output_count,
        "available_after_source_ns": available_after_ns,
        "accepted_heading_correction_count": len(corrections) - 1,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
