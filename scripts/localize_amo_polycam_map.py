#!/usr/bin/env python3
"""Causally initialize a pelvis-local trajectory against a Polycam room map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from g1_root_state_bridge.amo_dataset import load_lowstate
from g1_root_state_bridge.amo_map_localization import (
    global_localize,
    load_scan_archive,
    pelvis_structural_cloud,
)


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
    parser.add_argument("--map-key", default="map_xy_structural_5cm")
    parser.add_argument("--query-scans", type=Path, required=True)
    parser.add_argument("--query-treatments", type=Path, required=True)
    parser.add_argument("--query-treatment", required=True)
    parser.add_argument("--lowstate", type=Path, required=True)
    parser.add_argument("--query-start-source-ns", type=int, required=True)
    parser.add_argument("--query-duration-sec", type=float, default=10.0)
    parser.add_argument("--maximum-correspondence-m", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cloud-output", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.output, args.cloud_output):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    with np.load(args.map, allow_pickle=False) as archive:
        if args.map_key not in archive.files:
            raise ValueError(f"map key {args.map_key!r} not found")
        map_xy = np.asarray(archive[args.map_key], dtype=np.float64)
        map_metadata = json.loads(str(archive["metadata_json"]))
    query_end_ns = args.query_start_source_ns + round(args.query_duration_sec * 1e9)
    query_xy, query_audit = pelvis_structural_cloud(
        load_scan_archive(args.query_scans),
        _load_treatment(args.query_treatments, args.query_treatment),
        load_lowstate(args.lowstate),
        start_source_time_ns=args.query_start_source_ns,
        end_source_time_ns=query_end_ns,
        scan_stride=2,
    )
    report = global_localize(
        map_xy,
        query_xy,
        maximum_correspondence_m=args.maximum_correspondence_m,
    )
    report.update({
        "schema": "g1_polycam_map_localization_v1",
        "causal": True,
        "motive_online_input": False,
        "map_source_role": "candidate_polycam_prior_not_ground_truth",
        "map_path": str(args.map),
        "map_key": args.map_key,
        "map_metadata": map_metadata,
        "query_treatment": args.query_treatment,
        "query_audit": query_audit,
        "query_duration_sec": args.query_duration_sec,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        args.cloud_output,
        map_xy=map_xy.astype(np.float32),
        query_xy=query_xy.astype(np.float32),
        metadata_json=np.asarray(json.dumps({
            "schema": "g1_polycam_map_query_clouds_v1",
            "map_localization": str(args.output),
            "map_id": map_metadata["map_id"],
        }, sort_keys=True)),
    )
    print(json.dumps({
        "output": str(args.output),
        "healthy": bool(report["healthy"]),
        "yaw_deg": float(report["yaw_deg"]),
        "translation_m": report["translation_m"],
        "overlap_fraction": report["correlation"]["best_overlap_fraction"],
        "peak_ratio": report["correlation"]["peak_ratio"],
        "icp_inlier_fraction": report["icp"]["inlier_fraction"],
        "icp_p95_m": report["icp"]["p95_m"],
        "observable": report["icp"]["observability"]["observable"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
