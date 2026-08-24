"""Compare production replay poses against one frozen treatment track."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load_track(path: Path, treatment: str) -> dict[int, tuple[np.ndarray, float, int | None]]:
    result: dict[int, tuple[np.ndarray, float, int | None]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("kind") != "pose" or row.get("treatment") != treatment:
            continue
        time_ns = int(row["source_time_ns"])
        if time_ns in result:
            raise ValueError(f"duplicate {treatment} source timestamp: {time_ns}")
        position = np.asarray(row["position_xyz_m"], dtype=np.float64)
        quaternion = np.asarray(row["quaternion_xyzw"], dtype=np.float64)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("pose fields have the wrong shape")
        x, y, z, w = quaternion
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        sequence = row.get("source_lowstate_sequence")
        result[time_ns] = (
            position,
            yaw,
            None if sequence is None else int(sequence),
        )
    if not result:
        raise ValueError(f"no pose rows found for treatment {treatment!r}")
    return result


def compare(args: argparse.Namespace) -> dict[str, object]:
    candidate = _load_track(args.candidate, args.candidate_treatment)
    reference = _load_track(args.reference, args.reference_treatment)
    common = sorted(candidate.keys() & reference.keys())
    if len(common) < 3:
        raise ValueError("fewer than three exact source timestamps overlap")
    candidate_position = np.asarray([candidate[time_ns][0] for time_ns in common])
    reference_position = np.asarray([reference[time_ns][0] for time_ns in common])
    candidate_yaw = np.asarray([candidate[time_ns][1] for time_ns in common])
    reference_yaw = np.asarray([reference[time_ns][1] for time_ns in common])
    raw_delta_position = candidate_position - reference_position
    raw_delta_yaw = np.asarray(
        [
            math.atan2(
                math.sin(candidate[time_ns][1] - reference[time_ns][1]),
                math.cos(candidate[time_ns][1] - reference[time_ns][1]),
            )
            for time_ns in common
        ]
    )
    pose_comparison = getattr(args, "pose_comparison", "raw")
    if pose_comparison == "initial-relative":
        alignment_yaw = float(reference_yaw[0] - candidate_yaw[0])
        c, s = math.cos(alignment_yaw), math.sin(alignment_yaw)
        rotation = np.asarray(((c, -s), (s, c)), dtype=np.float64)
        candidate_relative_xy = (
            candidate_position[:, :2] - candidate_position[0, :2]
        ) @ rotation.T
        reference_relative_xy = reference_position[:, :2] - reference_position[0, :2]
        delta_position = np.column_stack(
            (
                candidate_relative_xy - reference_relative_xy,
                (candidate_position[:, 2] - candidate_position[0, 2])
                - (reference_position[:, 2] - reference_position[0, 2]),
            )
        )
        delta_yaw = np.asarray(
            [
                math.atan2(
                    math.sin((left - candidate_yaw[0]) - (right - reference_yaw[0])),
                    math.cos((left - candidate_yaw[0]) - (right - reference_yaw[0])),
                )
                for left, right in zip(candidate_yaw, reference_yaw)
            ]
        )
    elif pose_comparison == "raw":
        delta_position = raw_delta_position
        delta_yaw = raw_delta_yaw
    else:
        raise ValueError(f"unsupported pose comparison: {pose_comparison}")
    sequence_pairs = [
        (candidate[time_ns][2], reference[time_ns][2])
        for time_ns in common
        if candidate[time_ns][2] is not None and reference[time_ns][2] is not None
    ]
    coverage = len(common) / len(reference)
    planar_rmse_m = float(np.sqrt(np.mean(np.sum(delta_position[:, :2] ** 2, axis=1))))
    yaw_rmse_deg = float(np.degrees(np.sqrt(np.mean(delta_yaw**2))))
    raw_planar_rmse_m = float(
        np.sqrt(np.mean(np.sum(raw_delta_position[:, :2] ** 2, axis=1)))
    )
    raw_yaw_rmse_deg = float(np.degrees(np.sqrt(np.mean(raw_delta_yaw**2))))
    passed = (
        coverage >= args.minimum_reference_coverage
        and planar_rmse_m <= args.maximum_planar_rmse_m
        and yaw_rmse_deg <= args.maximum_yaw_rmse_deg
    )
    return {
        "schema": "g1_kiss_production_reference_comparison_v1",
        "status": "pass" if passed else "fail",
        "candidate": str(args.candidate.resolve()),
        "candidate_treatment": args.candidate_treatment,
        "reference": str(args.reference.resolve()),
        "reference_treatment": args.reference_treatment,
        "candidate_samples": len(candidate),
        "reference_samples": len(reference),
        "exact_timestamp_overlap": len(common),
        "reference_coverage": coverage,
        "pose_comparison": pose_comparison,
        "planar_rmse_m": planar_rmse_m,
        "raw_planar_rmse_m": raw_planar_rmse_m,
        "position_maximum_absolute_m": float(np.max(np.abs(delta_position))),
        "yaw_rmse_deg": yaw_rmse_deg,
        "raw_yaw_rmse_deg": raw_yaw_rmse_deg,
        "yaw_maximum_absolute_deg": float(np.degrees(np.max(np.abs(delta_yaw)))),
        "source_joint_sequence_match_fraction": (
            sum(left == right for left, right in sequence_pairs) / len(sequence_pairs)
            if sequence_pairs
            else None
        ),
        "thresholds": {
            "minimum_reference_coverage": args.minimum_reference_coverage,
            "maximum_planar_rmse_m": args.maximum_planar_rmse_m,
            "maximum_yaw_rmse_deg": args.maximum_yaw_rmse_deg,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-treatment", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-treatment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-reference-coverage", type=float, default=0.995)
    parser.add_argument("--maximum-planar-rmse-m", type=float, default=0.01)
    parser.add_argument("--maximum-yaw-rmse-deg", type=float, default=0.5)
    parser.add_argument(
        "--pose-comparison",
        choices=("raw", "initial-relative"),
        default="raw",
        help="compare absolute local poses or trajectories normalized at their first overlap",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
