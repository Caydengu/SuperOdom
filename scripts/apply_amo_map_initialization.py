#!/usr/bin/env python3
"""Add a causally available map-initialized pelvis treatment to a track."""

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
    parser.add_argument("--map-localization", type=Path, required=True)
    parser.add_argument("--source-treatment", default="superodom_dynamic_fk_pelvis")
    parser.add_argument(
        "--output-treatment",
        default="superodom_dynamic_fk_pelvis_fresh_map_init",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--map-corrections",
        type=Path,
        help="Optional causal sequence of accepted/rejected map corrections.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    localization = json.loads(args.map_localization.read_text(encoding="utf-8"))
    if not localization.get("healthy"):
        raise ValueError("map localization failed its frozen confidence gates")
    rotation_xy = np.asarray(localization["rotation_matrix"], dtype=np.float64)
    translation_xy = np.asarray(localization["translation_m"], dtype=np.float64)
    available_after_source_ns = int(localization["query_audit"]["end_source_time_ns"])
    angle = math.atan2(float(rotation_xy[0, 1]), float(rotation_xy[0, 0]))
    cosine, sine = math.cos(angle), math.sin(angle)
    world_rotation = np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    corrections = None
    if args.map_corrections is not None:
        correction_report = json.loads(args.map_corrections.read_text(encoding="utf-8"))
        if not correction_report.get("causal") or correction_report.get(
            "motive_online_input"
        ):
            raise ValueError(
                "map correction sequence violates the causal onboard-only contract"
            )
        corrections = [
            row for row in correction_report["corrections"] if row["accepted"]
        ]
        if not corrections:
            raise ValueError("map correction sequence contains no accepted transform")
    source_count = 0
    output_count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.treatments.open("r", encoding="utf-8") as source,
        args.output.open("x", encoding="utf-8") as output,
    ):
        for line in source:
            payload = json.loads(line)
            if payload.get("kind") == "metadata":
                payload["map_localization"] = {
                    "path": str(args.map_localization),
                    "healthy": True,
                    "motive_online_input": False,
                    "available_after_source_ns": available_after_source_ns,
                    "source_treatment": args.source_treatment,
                    "output_treatment": args.output_treatment,
                    "corrections_path": (
                        str(args.map_corrections) if args.map_corrections else None
                    ),
                }
            output.write(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
            )
            if payload.get("treatment") != args.source_treatment:
                continue
            source_count += 1
            if int(payload["source_time_ns"]) < available_after_source_ns:
                continue
            correction_index = 0
            if corrections is not None:
                source_time_ns = int(payload["source_time_ns"])
                correction_index = max(
                    index
                    for index, row in enumerate(corrections)
                    if int(row["available_after_source_ns"]) <= source_time_ns
                )
                active = corrections[correction_index]
                rotation_xy = np.asarray(active["rotation_matrix"], dtype=np.float64)
                translation_xy = np.asarray(active["translation_m"], dtype=np.float64)
                angle = math.atan2(float(rotation_xy[0, 1]), float(rotation_xy[0, 0]))
                cosine, sine = math.cos(angle), math.sin(angle)
                world_rotation = np.asarray(
                    ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
                    dtype=np.float64,
                )
            corrected = dict(payload)
            corrected["treatment"] = args.output_treatment
            corrected["map_initialized"] = True
            corrected["map_initialization_source_time_ns"] = available_after_source_ns
            corrected["map_correction_index"] = correction_index
            position = np.asarray(payload["position_xyz_m"], dtype=np.float64)
            position[:2] = position[:2] @ rotation_xy + translation_xy
            corrected["position_xyz_m"] = position.tolist()
            corrected["quaternion_xyzw"] = list(
                xyzw_from_rotation(
                    world_rotation
                    @ rotation_from_xyzw(np.asarray(payload["quaternion_xyzw"]))
                )
            )
            output.write(
                json.dumps(corrected, sort_keys=True, separators=(",", ":")) + "\n"
            )
            output_count += 1
    if output_count == 0:
        raise ValueError("no source treatment records occur after map initialization")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_count": source_count,
                "map_initialized_count": output_count,
                "available_after_source_ns": available_after_source_ns,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
