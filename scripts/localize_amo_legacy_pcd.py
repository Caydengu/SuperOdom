#!/usr/bin/env python3
"""Localize a causal AMO query cloud against an unchanged legacy binary PCD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from g1_root_state_bridge.amo_map_localization import global_localize, voxelize_xy


def load_binary_xy(path: Path, *, minimum_z_m: float, maximum_z_m: float) -> np.ndarray:
    payload = path.read_bytes()
    marker = b"DATA binary\n"
    offset = payload.index(marker) + len(marker)
    header = payload[:offset].decode("ascii")
    fields = next(
        line for line in header.splitlines() if line.startswith("FIELDS ")
    ).split()[1:]
    sizes = [
        int(value)
        for value in next(
            line for line in header.splitlines() if line.startswith("SIZE ")
        ).split()[1:]
    ]
    types = next(
        line for line in header.splitlines() if line.startswith("TYPE ")
    ).split()[1:]
    if (
        fields[:3] != ["x", "y", "z"]
        or sizes[:3] != [4, 4, 4]
        or types[:3] != ["F", "F", "F"]
    ):
        raise ValueError("legacy PCD is not little-endian float32 x/y/z")
    dtype = np.dtype([(name, "<f4") for name in fields])
    records = np.frombuffer(payload, dtype=dtype, offset=offset)
    xyz = np.column_stack((records["x"], records["y"], records["z"]))
    admitted = (
        np.all(np.isfinite(xyz), axis=1)
        & (xyz[:, 2] >= minimum_z_m)
        & (xyz[:, 2] <= maximum_z_m)
    )
    return voxelize_xy(xyz[admitted, :2], 0.10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--query-clouds", type=Path, required=True)
    parser.add_argument("--query-audit-from", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cloud-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.cloud_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    map_xy = load_binary_xy(args.pcd, minimum_z_m=0.15, maximum_z_m=2.85)
    with np.load(args.query_clouds, allow_pickle=False) as clouds:
        query_xy = np.asarray(clouds["query_xy"], dtype=np.float64)
    query_source = json.loads(args.query_audit_from.read_text(encoding="utf-8"))
    report = global_localize(map_xy, query_xy)
    report.update(
        {
            "map_source_role": "legacy_field_bay_scan_unchanged",
            "map_path": str(args.pcd),
            "map_point_count": int(map_xy.shape[0]),
            "query_audit": query_source["query_audit"],
            "query_duration_sec": query_source["query_duration_sec"],
            "motive_online_input": False,
            "causal": True,
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
