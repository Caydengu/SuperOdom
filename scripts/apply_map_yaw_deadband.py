#!/usr/bin/env python3
"""Convert small accepted map-yaw innovations into confidence-only refreshes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _shortest_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _yaw_deg(rotation: np.ndarray) -> float:
    return math.degrees(math.atan2(float(rotation[0, 1]), float(rotation[0, 0])))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--minimum-yaw-update-deg", type=float, required=True)
    parser.add_argument("--yaw-gain", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.minimum_yaw_update_deg < 0.0:
        raise ValueError("minimum yaw update must be nonnegative")
    if not 0.0 <= args.yaw_gain <= 1.0:
        raise ValueError("yaw gain must lie in [0, 1]")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    if report.get("motive_online_input") is not False:
        raise ValueError("map correction report must be onboard-only")
    corrections = report.get("corrections", [])
    if not corrections or corrections[0].get("kind") != "global_initialization":
        raise ValueError("first accepted correction must be global initialization")

    active_rotation = np.asarray(corrections[0]["rotation_matrix"], dtype=np.float64)
    active_translation = np.asarray(corrections[0]["translation_m"], dtype=np.float64)
    output_rows: list[dict[str, object]] = []
    applied_update_count = 0
    confidence_refresh_count = 0
    for index, source_row in enumerate(corrections):
        row = dict(source_row)
        quality = dict(row.get("quality", {}))
        row["quality"] = quality
        if index == 0 or not bool(row.get("accepted")):
            output_rows.append(row)
            continue
        candidate_rotation = np.asarray(row["rotation_matrix"], dtype=np.float64)
        candidate_translation = np.asarray(row["translation_m"], dtype=np.float64)
        active_yaw = _yaw_deg(active_rotation)
        candidate_yaw = _yaw_deg(candidate_rotation)
        innovation_deg = _shortest_degrees(candidate_yaw - active_yaw)
        quality["raw_yaw_innovation_deg"] = innovation_deg
        quality["raw_rotation_matrix"] = candidate_rotation.tolist()
        quality["raw_translation_m"] = candidate_translation.tolist()
        if abs(innovation_deg) < args.minimum_yaw_update_deg:
            row["rotation_matrix"] = active_rotation.tolist()
            row["translation_m"] = active_translation.tolist()
            row["yaw_deg"] = active_yaw
            row["application"] = "confidence_only_refresh"
            confidence_refresh_count += 1
        else:
            applied_yaw = active_yaw + args.yaw_gain * innovation_deg
            angle = math.radians(applied_yaw)
            cosine, sine = math.cos(angle), math.sin(angle)
            applied_rotation = np.asarray(
                ((cosine, sine), (-sine, cosine)), dtype=np.float64
            )
            row["rotation_matrix"] = applied_rotation.tolist()
            row["yaw_deg"] = applied_yaw
            quality["yaw_gain"] = args.yaw_gain
            row["application"] = "heading_update"
            active_rotation = applied_rotation
            active_translation = candidate_translation
            applied_update_count += 1
        output_rows.append(row)

    result = dict(report)
    result.update(
        {
            "schema": "g1_polycam_deadbanded_map_corrections_v1",
            "source_report": str(args.input),
            "minimum_yaw_update_deg": args.minimum_yaw_update_deg,
            "yaw_gain": args.yaw_gain,
            "accepted_heading_update_count": applied_update_count,
            "confidence_only_refresh_count": confidence_refresh_count,
            "corrections": output_rows,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted_heading_update_count": applied_update_count,
                "confidence_only_refresh_count": confidence_refresh_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
