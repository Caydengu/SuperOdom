#!/usr/bin/env python3
"""Cross-score independent Polycam map localizations without online ground truth.

The input manifest names one onboard-only map-localization result and one local
odometry treatment for each run.  Motive is used only after inference: a fixed
Motive-to-map transform is fitted on one run and then applied unchanged to every
other run.  This distinguishes a repeatable map placement from a visually good
but globally ambiguous ICP match.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from g1_root_state_bridge.amo_dataset import load_motive
from g1_root_state_bridge.amo_scoring import (
    _interpolate_reference,
    _robotics_yaw_z_up,
    provisional_reference,
)


def _shortest(angle: np.ndarray | float) -> np.ndarray:
    values = np.asarray(angle, dtype=np.float64)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def _circular_mean(angle: np.ndarray) -> float:
    return float(np.arctan2(np.mean(np.sin(angle)), np.mean(np.cos(angle))))


def _fit_rigid_row(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    u, _, vh = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    translation = target_mean - source_mean @ rotation
    return rotation, translation


def _read_treatment(path: Path, treatment: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            if payload.get("treatment") == treatment:
                records.append(payload)
    if len(records) < 2:
        raise ValueError(f"{path} contains fewer than two {treatment!r} records")
    return records


def _load_run(
    spec: dict[str, Any],
    *,
    front_plane_to_pelvis_x_m: float,
    motive_rigid_body_id: int,
    motive_rigid_body_name: str,
) -> dict[str, Any]:
    records = _read_treatment(Path(spec["treatments"]), str(spec["treatment"]))
    localization = json.loads(Path(spec["localization"]).read_text(encoding="utf-8"))
    if not localization.get("healthy"):
        raise ValueError(f"map localization is unhealthy for {spec['name']!r}")
    if localization.get("motive_online_input") is not False:
        raise ValueError(f"map localization is not onboard-only for {spec['name']!r}")

    event = np.asarray([row["event_realtime_ns"] for row in records], dtype=np.int64)
    local_position = np.asarray(
        [row["position_xyz_m"] for row in records], dtype=np.float64
    )
    local_quaternion = np.asarray(
        [row["quaternion_xyzw"] for row in records], dtype=np.float64
    )
    map_rotation = np.asarray(localization["rotation_matrix"], dtype=np.float64)
    map_translation = np.asarray(localization["translation_m"], dtype=np.float64)
    map_position = local_position[:, :2] @ map_rotation + map_translation
    map_yaw = _robotics_yaw_z_up(local_quaternion) + math.atan2(
        float(map_rotation[0, 1]), float(map_rotation[0, 0])
    )

    motive = load_motive(
        Path(spec["motive"]),
        expected_rigid_body_id=motive_rigid_body_id,
        expected_rigid_body_name=motive_rigid_body_name,
    )
    reference_time, reference_position, reference_yaw = provisional_reference(
        motive, front_plane_to_pelvis_x_m=front_plane_to_pelvis_x_m
    )
    overlap = (event >= reference_time[0]) & (event <= reference_time[-1])
    event = event[overlap]
    map_position = map_position[overlap]
    map_yaw = map_yaw[overlap]
    reference_position, reference_yaw = _interpolate_reference(
        reference_time, reference_position, reference_yaw, event
    )
    motive_xy = np.column_stack((reference_position[:, 0], -reference_position[:, 2]))
    available_after_ns = int(localization["query_audit"]["end_source_time_ns"])
    fit_start_ns = int(spec["fit_start_ns"])
    return {
        "name": str(spec["name"]),
        "event_time_ns": event,
        "map_position_xy": map_position,
        "map_yaw_rad": map_yaw,
        "motive_position_xy": motive_xy,
        "motive_yaw_rad": reference_yaw,
        "fit_start_ns": fit_start_ns,
        "available_after_ns": available_after_ns,
        "initialization": localization,
    }


def _select(run: dict[str, Any], admitted: np.ndarray) -> dict[str, Any]:
    return {
        key: value[admitted] if isinstance(value, np.ndarray) else value
        for key, value in run.items()
    }


def _fit_transform(run: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    rotation, translation = _fit_rigid_row(
        np.asarray(run["motive_position_xy"]), np.asarray(run["map_position_xy"])
    )
    yaw = _circular_mean(
        _shortest(np.asarray(run["map_yaw_rad"]) - np.asarray(run["motive_yaw_rad"]))
    )
    return rotation, translation, yaw


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def _cross_score(
    train: dict[str, Any], test: dict[str, Any]
) -> dict[str, Any]:
    rotation, translation, yaw = _fit_transform(train)
    predicted_position = np.asarray(test["motive_position_xy"]) @ rotation + translation
    position_error = np.linalg.norm(
        predicted_position - np.asarray(test["map_position_xy"]), axis=1
    )
    yaw_error_deg = np.degrees(
        np.abs(
            _shortest(
                np.asarray(test["motive_yaw_rad"]) + yaw
                - np.asarray(test["map_yaw_rad"])
            )
        )
    )
    return {
        "sample_count": int(position_error.size),
        "position_error_m": _distribution(position_error),
        "yaw_error_deg": _distribution(yaw_error_deg),
        "trained_motive_to_map_rotation_deg": float(
            math.degrees(math.atan2(rotation[0, 1], rotation[0, 0]))
        ),
        "trained_motive_to_map_translation_m": translation.tolist(),
        "trained_motive_to_map_heading_offset_deg": math.degrees(yaw),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-manifest", type=Path, required=True)
    parser.add_argument("--fit-duration-sec", type=float, default=10.0)
    parser.add_argument("--front-plane-to-pelvis-x-m", type=float, default=0.0)
    parser.add_argument("--motive-rigid-body-id", type=int, default=42)
    parser.add_argument("--motive-rigid-body-name", default="G1_PELVIS_F_4123")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    manifest = json.loads(args.runs_manifest.read_text(encoding="utf-8"))
    specs = manifest.get("runs", [])
    if len(specs) < 2:
        raise ValueError("runs manifest requires at least two independent runs")
    names = [str(spec["name"]) for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("run names must be unique")

    full_runs: dict[str, dict[str, Any]] = {}
    fit_runs: dict[str, dict[str, Any]] = {}
    causal_runs: dict[str, dict[str, Any]] = {}
    for spec in specs:
        run = _load_run(
            spec,
            front_plane_to_pelvis_x_m=args.front_plane_to_pelvis_x_m,
            motive_rigid_body_id=args.motive_rigid_body_id,
            motive_rigid_body_name=args.motive_rigid_body_name,
        )
        name = str(run["name"])
        event = np.asarray(run["event_time_ns"])
        fit_end_ns = int(run["fit_start_ns"]) + round(args.fit_duration_sec * 1e9)
        fit_mask = (event >= int(run["fit_start_ns"])) & (event <= fit_end_ns)
        causal_mask = event >= int(run["available_after_ns"])
        if np.sum(fit_mask) < 3:
            raise ValueError(f"fit window contains fewer than three samples for {name!r}")
        if np.sum(causal_mask) < 3:
            raise ValueError(f"causal map window contains fewer than three samples for {name!r}")
        full_runs[name] = run
        fit_runs[name] = _select(run, fit_mask)
        causal_runs[name] = _select(run, causal_mask)

    pairwise: dict[str, dict[str, Any]] = {}
    for train_name in names:
        for test_name in names:
            if train_name == test_name:
                continue
            pairwise[f"{train_name}_to_{test_name}"] = {
                "initial_window": _cross_score(fit_runs[train_name], fit_runs[test_name]),
                "causal_post_initialization": _cross_score(
                    fit_runs[train_name], causal_runs[test_name]
                ),
                "diagnostic_full_run": _cross_score(
                    fit_runs[train_name], full_runs[test_name]
                ),
            }

    fitted = {name: _fit_transform(fit_runs[name]) for name in names}
    transform_deltas: dict[str, dict[str, float]] = {}
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1 :]:
            first = fitted[first_name]
            second = fitted[second_name]
            first_angle = math.atan2(first[0][0, 1], first[0][0, 0])
            second_angle = math.atan2(second[0][0, 1], second[0][0, 0])
            transform_deltas[f"{first_name}_vs_{second_name}"] = {
                "rotation_delta_deg": float(
                    math.degrees(_shortest(second_angle - first_angle))
                ),
                "translation_delta_m": float(np.linalg.norm(second[1] - first[1])),
                "heading_offset_delta_deg": float(
                    math.degrees(_shortest(second[2] - first[2]))
                ),
            }

    report = {
        "schema": "g1_polycam_multi_run_consistency_v1",
        "causal": True,
        "motive_role": "evaluator_only",
        "motive_online_input": False,
        "fit_duration_sec": args.fit_duration_sec,
        "front_plane_to_pelvis_x_m": args.front_plane_to_pelvis_x_m,
        "run_names": names,
        "pairwise": pairwise,
        "independent_motive_to_map_transform_deltas": transform_deltas,
        "interpretation_contract": (
            "Cross-run consistency can reject a false map placement. It does not make "
            "the Polycam mesh, map frame, or Motive-to-pelvis lever arm surveyed truth."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "pair_count": len(pairwise)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
