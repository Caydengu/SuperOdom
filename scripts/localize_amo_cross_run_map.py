#!/usr/bin/env python3
"""Build a development structural map and localize a causal holdout window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from g1_root_state_bridge.amo_map_localization import (
    global_localize,
    load_scan_archive,
    structural_cloud,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-scans", type=Path, required=True)
    parser.add_argument("--map-odometry", type=Path, required=True)
    parser.add_argument("--map-start-source-ns", type=int, required=True)
    parser.add_argument("--map-end-source-ns", type=int, required=True)
    parser.add_argument("--query-scans", type=Path, required=True)
    parser.add_argument("--query-odometry", type=Path, required=True)
    parser.add_argument("--query-start-source-ns", type=int, required=True)
    parser.add_argument("--query-duration-sec", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cloud-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.cloud_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    map_xy, map_audit = structural_cloud(
        load_scan_archive(args.map_scans),
        args.map_odometry,
        start_source_time_ns=args.map_start_source_ns,
        end_source_time_ns=args.map_end_source_ns,
        scan_stride=5,
    )
    query_end = args.query_start_source_ns + round(args.query_duration_sec * 1e9)
    query_xy, query_audit = structural_cloud(
        load_scan_archive(args.query_scans),
        args.query_odometry,
        start_source_time_ns=args.query_start_source_ns,
        end_source_time_ns=query_end,
        scan_stride=2,
    )
    report = global_localize(map_xy, query_xy)
    report.update(
        {
            "map_audit": map_audit,
            "query_audit": query_audit,
            "query_duration_sec": args.query_duration_sec,
            "motive_online_input": False,
            "causal": True,
            "map_source_role": "development_walk_only",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(args.cloud_output, map_xy=map_xy, query_xy=query_xy)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "healthy": report["healthy"],
                "yaw_deg": report["yaw_deg"],
                "translation_m": report["translation_m"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
