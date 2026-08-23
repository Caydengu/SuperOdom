#!/usr/bin/env python3
"""Materialize an exact-time, rotationally deskewed Livox scan archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_kiss_icp_scan_archive import gyro_deskew_scan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gyro-bias-radps", nargs=3, type=float, required=True)
    parser.add_argument("--reference", choices=("start", "end"), default="end")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    archive = np.load(args.input, allow_pickle=False)
    points = np.asarray(archive["points_xyz_m"], dtype=np.float32)
    offsets = np.asarray(archive["scan_offsets"], dtype=np.int64)
    source_time_ns = np.asarray(archive["source_time_ns"], dtype=np.int64)
    relative_time_s = np.asarray(archive["point_relative_time_s"], dtype=np.float32)
    imu_time_ns = np.asarray(archive["imu_source_time_ns"], dtype=np.int64)
    gyro = np.asarray(archive["imu_angular_velocity_radps"], dtype=np.float64)
    output_points = np.empty_like(points)
    output_time_ns = source_time_ns.copy()
    valid = np.ones(source_time_ns.size, dtype=bool)
    excursion_deg = np.zeros(source_time_ns.size, dtype=np.float32)
    reasons: dict[str, int] = {}
    for index in range(source_time_ns.size):
        selected = slice(offsets[index], offsets[index + 1])
        try:
            deskewed, output_time_ns[index], excursion_deg[index] = gyro_deskew_scan(
                points[selected],
                relative_time_s[selected],
                scan_start_time_ns=int(source_time_ns[index]),
                imu_time_ns=imu_time_ns,
                imu_angular_velocity_radps=gyro,
                gyro_bias_radps=np.asarray(args.gyro_bias_radps, dtype=np.float64),
                reference=args.reference,
            )
            output_points[selected] = deskewed.astype(np.float32)
        except ValueError as error:
            reason = str(error)
            if not reason.startswith(("IMU begins", "IMU ends")):
                raise
            output_points[selected] = points[selected]
            valid[index] = False
            reasons[reason] = reasons.get(reason, 0) + 1
    source_metadata = json.loads(str(archive["metadata_json"]))
    metadata = {
        **source_metadata,
        "schema": "g1_livox_rotationally_deskewed_scan_archive_v1",
        "source_archive": str(args.input),
        "reference": args.reference,
        "gyro_bias_radps": list(args.gyro_bias_radps),
        "deskew_valid_fraction": float(np.mean(valid)),
        "deskew_failure_reasons": reasons,
        "translation_deskew": "disabled",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points_xyz_m=output_points,
        scan_offsets=offsets,
        source_time_ns=output_time_ns,
        receipt_time_ns=np.asarray(archive["receipt_time_ns"], dtype=np.int64),
        deskew_valid=valid,
        deskew_angular_excursion_deg=excursion_deg,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps({
        "output": str(args.output),
        "scan_count": int(source_time_ns.size),
        "point_count": int(points.shape[0]),
        "deskew_valid_fraction": float(np.mean(valid)),
        "failure_reasons": reasons,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
