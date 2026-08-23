#!/usr/bin/env python3
"""Measure whether map-ICP translation innovations track true local drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from g1_root_state_bridge.amo_dataset import load_motive
from g1_root_state_bridge.amo_scoring import _interpolate_reference, provisional_reference


def _load(path: Path, treatment: str) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("treatment") == treatment:
                result.append(row)
    if len(result) < 2:
        raise ValueError(f"missing treatment {treatment!r}")
    return result


def _fit_rigid_row(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    u, _, vh = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation, target_mean - source_mean @ rotation


def _nearest(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.clip(np.searchsorted(reference, query, side="left"), 0, reference.size - 1)
    left = np.clip(right - 1, 0, reference.size - 1)
    return np.where(
        np.abs(query - reference[left]) <= np.abs(reference[right] - query), left, right
    )


def _transform(row: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(row["rotation_matrix"], dtype=np.float64),
        np.asarray(row["translation_m"], dtype=np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatments", type=Path, required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--motive", type=Path, required=True)
    parser.add_argument("--fit-start-ns", type=int, required=True)
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
    local_xy = np.asarray([row["position_xyz_m"][:2] for row in records], dtype=np.float64)
    report = json.loads(args.corrections.read_text(encoding="utf-8"))
    corrections = [row for row in report["corrections"] if row["accepted"]]
    initial_rotation, initial_translation = _transform(corrections[0])
    map_fixed = local_xy @ initial_rotation + initial_translation

    motive = load_motive(
        args.motive,
        expected_rigid_body_id=args.motive_rigid_body_id,
        expected_rigid_body_name=args.motive_rigid_body_name,
    )
    reference_time, reference_position, _ = provisional_reference(
        motive,
        front_plane_to_pelvis_x_m=args.front_plane_to_pelvis_x_m,
    )
    valid = (time_ns >= reference_time[0]) & (time_ns <= reference_time[-1])
    time_ns = time_ns[valid]
    local_xy = local_xy[valid]
    map_fixed = map_fixed[valid]
    reference_position, _ = _interpolate_reference(
        reference_time, reference_position, np.zeros(reference_time.size), time_ns
    )
    motive_xy = np.column_stack((reference_position[:, 0], -reference_position[:, 2]))
    fit = (time_ns >= args.fit_start_ns) & (
        time_ns <= args.fit_start_ns + round(args.fit_duration_sec * 1e9)
    )
    motive_rotation, motive_translation = _fit_rigid_row(motive_xy[fit], map_fixed[fit])
    reference_map_xy = motive_xy @ motive_rotation + motive_translation

    correction_time = np.asarray(
        [row["available_after_source_ns"] for row in corrections], dtype=np.int64
    )
    within = (correction_time >= time_ns[0]) & (correction_time <= time_ns[-1])
    corrections = [row for row, keep in zip(corrections, within) if keep]
    correction_time = correction_time[within]
    index = _nearest(time_ns, correction_time)
    fixed_at_update = map_fixed[index]
    reference_at_update = reference_map_xy[index]
    candidate_at_update: list[np.ndarray] = []
    features: list[dict[str, object]] = []
    for row, local_index in zip(corrections, index):
        rotation, translation = _transform(row)
        candidate_at_update.append(local_xy[local_index] @ rotation + translation)
        quality = row["quality"]
        observability = quality["observability"]
        features.append(
            {
                "available_after_source_ns": int(row["available_after_source_ns"]),
                "inlier_fraction": float(quality["inlier_fraction"]),
                "rmse_m": float(quality["rmse_m"]),
                "p95_m": float(quality["p95_m"]),
                "minimum_eigenvalue": float(observability["minimum_eigenvalue"]),
                "condition_number": float(observability["condition_number"]),
            }
        )
    candidate_at_update = np.stack(candidate_at_update)
    innovation = candidate_at_update - fixed_at_update
    ideal = reference_at_update - fixed_at_update
    fixed_error = np.linalg.norm(ideal, axis=1)
    candidate_error = np.linalg.norm(candidate_at_update - reference_at_update, axis=1)
    innovation_norm = np.linalg.norm(innovation, axis=1)
    ideal_norm = np.linalg.norm(ideal, axis=1)
    cosine = np.sum(innovation * ideal, axis=1) / np.maximum(
        1e-9, innovation_norm * ideal_norm
    )
    for row, values, target, before, after, similarity in zip(
        features, innovation, ideal, fixed_error, candidate_error, cosine
    ):
        row.update(
            innovation_m=values.tolist(),
            ideal_correction_m=target.tolist(),
            fixed_error_m=float(before),
            candidate_error_m=float(after),
            improves=bool(after < before),
            directional_cosine=float(similarity),
        )
    component_correlation = [
        float(np.corrcoef(innovation[:, axis], ideal[:, axis])[0, 1])
        for axis in range(2)
    ]
    flattened_correlation = float(
        np.corrcoef(innovation.reshape(-1), ideal.reshape(-1))[0, 1]
    )
    output = {
        "schema": "g1_polycam_translation_signal_audit_v1",
        "motive_role": "evaluator_only_discovery_diagnostic",
        "motive_online_input": False,
        "treatment": args.treatment,
        "correction_count": len(corrections),
        "raw_candidate_improves_fraction": float(np.mean(candidate_error < fixed_error)),
        "fixed_position_rmse_at_updates_m": float(np.sqrt(np.mean(fixed_error**2))),
        "candidate_position_rmse_at_updates_m": float(
            np.sqrt(np.mean(candidate_error**2))
        ),
        "translation_signal": {
            "component_correlation_xy": component_correlation,
            "flattened_correlation": flattened_correlation,
            "directional_cosine_median": float(np.median(cosine)),
            "directional_cosine_positive_fraction": float(np.mean(cosine > 0.0)),
            "innovation_norm_median_m": float(np.median(innovation_norm)),
            "ideal_norm_median_m": float(np.median(ideal_norm)),
        },
        "rows": features,
        "interpretation_contract": (
            "A useful translation lane must predict the direction of evaluator-only "
            "accumulated drift; ICP residual quality alone is not sufficient"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in output.items() if key != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
