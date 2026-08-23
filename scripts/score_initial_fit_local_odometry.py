#!/usr/bin/env python3
"""Score accumulated local-odometry drift after a causal initial alignment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from g1_root_state_bridge.amo_dataset import load_motive
from g1_root_state_bridge.amo_scoring import (
    _interpolate_reference,
    _robotics_yaw_z_up,
    provisional_reference,
)


def _load(path: Path, treatment: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("treatment") == treatment:
                records.append(row)
    if len(records) < 2:
        raise ValueError(f"missing treatment {treatment!r}")
    return records


def _fit_rigid_row(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    u, _, vh = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation, target_mean - source_mean @ rotation


def _shortest(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatments", type=Path, required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--motive", type=Path, required=True)
    parser.add_argument("--start-source-ns", type=int, required=True)
    parser.add_argument("--fit-duration-sec", type=float, default=10.0)
    parser.add_argument("--front-plane-to-pelvis-x-m", type=float, default=0.0)
    parser.add_argument("--motive-rigid-body-id", type=int, default=42)
    parser.add_argument("--motive-rigid-body-name", default="G1_PELVIS_F_4123")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    records = _load(args.treatments, args.treatment)
    time_ns = np.asarray([row["source_time_ns"] for row in records], dtype=np.int64)
    position = np.asarray([row["position_xyz_m"] for row in records], dtype=np.float64)
    quaternion = np.asarray([row["quaternion_xyzw"] for row in records], dtype=np.float64)
    motive = load_motive(
        args.motive,
        expected_rigid_body_id=args.motive_rigid_body_id,
        expected_rigid_body_name=args.motive_rigid_body_name,
    )
    ref_time, ref_position, ref_yaw = provisional_reference(
        motive,
        front_plane_to_pelvis_x_m=args.front_plane_to_pelvis_x_m,
    )
    end_source_ns = min(int(time_ns[-1]), int(ref_time[-1]))
    admitted = (time_ns >= args.start_source_ns) & (time_ns <= end_source_ns)
    time_ns = time_ns[admitted]
    position = position[admitted]
    quaternion = quaternion[admitted]
    ref_position, ref_yaw = _interpolate_reference(
        ref_time, ref_position, ref_yaw, time_ns
    )
    motive_xy = np.column_stack((ref_position[:, 0], -ref_position[:, 2]))
    local_xy = position[:, :2]
    fit = time_ns <= args.start_source_ns + round(args.fit_duration_sec * 1e9)
    if np.sum(fit) < 10:
        raise ValueError("fewer than ten samples in the initial fit window")
    rotation, translation = _fit_rigid_row(local_xy[fit], motive_xy[fit])
    predicted_xy = local_xy @ rotation + translation
    position_error = np.linalg.norm(predicted_xy - motive_xy, axis=1)
    local_yaw = _robotics_yaw_z_up(quaternion)
    yaw_offset = float(
        np.arctan2(
            np.mean(np.sin(ref_yaw[fit] - local_yaw[fit])),
            np.mean(np.cos(ref_yaw[fit] - local_yaw[fit])),
        )
    )
    yaw_error_deg = np.degrees(np.abs(_shortest(local_yaw + yaw_offset - ref_yaw)))
    path_length_m = float(np.sum(np.linalg.norm(np.diff(motive_xy, axis=0), axis=1)))
    duration_s = float((time_ns[-1] - time_ns[0]) * 1e-9)
    report = {
        "schema": "g1_initial_fit_local_odometry_score_v1",
        "motive_role": "evaluator_only",
        "motive_online_input": False,
        "treatment": args.treatment,
        "front_plane_to_pelvis_x_m": args.front_plane_to_pelvis_x_m,
        "start_source_ns": args.start_source_ns,
        "fit_duration_sec": args.fit_duration_sec,
        "sample_count": int(time_ns.size),
        "duration_s": duration_s,
        "reference_path_length_m": path_length_m,
        "position": {
            "rmse_m": float(np.sqrt(np.mean(position_error**2))),
            "p95_m": float(np.quantile(position_error, 0.95)),
            "maximum_m": float(np.max(position_error)),
            "final_m": float(position_error[-1]),
            "error_per_path_length": float(np.sqrt(np.mean(position_error**2)) / path_length_m),
        },
        "yaw": {
            "rmse_deg": float(np.sqrt(np.mean(yaw_error_deg**2))),
            "p95_deg": float(np.quantile(yaw_error_deg, 0.95)),
            "maximum_deg": float(np.max(yaw_error_deg)),
        },
        "initial_alignment": {
            "rotation_deg": math.degrees(math.atan2(rotation[0, 1], rotation[0, 0])),
            "translation_m": translation.tolist(),
            "yaw_offset_deg": math.degrees(yaw_offset),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
