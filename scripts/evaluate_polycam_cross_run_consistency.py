#!/usr/bin/env python3
"""Evaluate independent map initializations in the shared Motive frame."""

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


def _fit_rigid_row(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    u, _, vh = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return rotation, target_mean - source_mean @ rotation


def _shortest(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _circular_mean(angle: np.ndarray) -> float:
    return float(np.arctan2(np.mean(np.sin(angle)), np.mean(np.cos(angle))))


def _load_run(
    treatments: Path,
    treatment: str,
    localization: Path,
    motive_path: Path,
    *,
    front_plane_to_pelvis_x_m: float,
    motive_rigid_body_id: int,
    motive_rigid_body_name: str,
    treatments_already_map: bool,
) -> dict[str, np.ndarray | dict[str, object]]:
    records: list[dict[str, object]] = []
    with treatments.open(encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            if payload.get("treatment") == treatment:
                records.append(payload)
    if len(records) < 2:
        raise ValueError(f"missing treatment {treatment!r}")
    initialization = json.loads(localization.read_text(encoding="utf-8"))
    if not initialization.get("healthy"):
        raise ValueError(f"map initialization is not healthy: {localization}")
    event = np.asarray([row["event_realtime_ns"] for row in records], dtype=np.int64)
    local_position = np.asarray([row["position_xyz_m"] for row in records], dtype=np.float64)
    local_quaternion = np.asarray([row["quaternion_xyzw"] for row in records], dtype=np.float64)
    if treatments_already_map:
        map_position = local_position[:, :2]
        map_yaw = _robotics_yaw_z_up(local_quaternion)
    else:
        map_rotation = np.asarray(initialization["rotation_matrix"], dtype=np.float64)
        map_translation = np.asarray(initialization["translation_m"], dtype=np.float64)
        map_position = local_position[:, :2] @ map_rotation + map_translation
        map_yaw = _robotics_yaw_z_up(local_quaternion) + math.atan2(
            float(map_rotation[0, 1]), float(map_rotation[0, 0])
        )
    motive = load_motive(
        motive_path,
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
    return {
        "event_time_ns": event,
        "map_position_xy": map_position,
        "map_yaw_rad": map_yaw,
        "motive_position_xy": motive_xy,
        "motive_yaw_rad": reference_yaw,
        "initialization": initialization,
    }


def _fit_transform(run: dict[str, object]) -> tuple[np.ndarray, np.ndarray, float]:
    rotation, translation = _fit_rigid_row(
        np.asarray(run["motive_position_xy"]), np.asarray(run["map_position_xy"])
    )
    yaw = _circular_mean(
        _shortest(np.asarray(run["map_yaw_rad"]) - np.asarray(run["motive_yaw_rad"]))
    )
    return rotation, translation, yaw


def _cross_score(
    train: dict[str, object], test: dict[str, object]
) -> dict[str, object]:
    rotation, translation, yaw = _fit_transform(train)
    predicted = np.asarray(test["motive_position_xy"]) @ rotation + translation
    error = np.linalg.norm(predicted - np.asarray(test["map_position_xy"]), axis=1)
    yaw_error = np.degrees(
        np.abs(
            _shortest(
                np.asarray(test["motive_yaw_rad"]) + yaw
                - np.asarray(test["map_yaw_rad"])
            )
        )
    )
    return {
        "sample_count": int(error.size),
        "position_rmse_m": float(np.sqrt(np.mean(error**2))),
        "position_p95_m": float(np.quantile(error, 0.95)),
        "yaw_rmse_deg": float(np.sqrt(np.mean(yaw_error**2))),
        "yaw_p95_deg": float(np.quantile(yaw_error, 0.95)),
        "trained_motive_to_map_rotation_deg": float(
            math.degrees(math.atan2(rotation[0, 1], rotation[0, 0]))
        ),
        "trained_motive_to_map_translation_m": translation.tolist(),
        "trained_motive_to_map_heading_offset_deg": math.degrees(yaw),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for run in ("walk02", "walk03"):
        parser.add_argument(f"--{run}-treatments", type=Path, required=True)
        parser.add_argument(f"--{run}-localization", type=Path, required=True)
        parser.add_argument(f"--{run}-motive", type=Path, required=True)
        parser.add_argument(f"--{run}-start-ns", type=int, required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--fit-duration-sec", type=float, default=10.0)
    parser.add_argument("--front-plane-to-pelvis-x-m", type=float, default=0.0)
    parser.add_argument("--motive-rigid-body-id", type=int, default=42)
    parser.add_argument("--motive-rigid-body-name", default="G1_PELVIS_F_4123")
    parser.add_argument("--treatments-already-map", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    full_runs: dict[str, dict[str, object]] = {}
    fit_runs: dict[str, dict[str, object]] = {}
    for name in ("walk02", "walk03"):
        full = _load_run(
            getattr(args, f"{name}_treatments"),
            args.treatment,
            getattr(args, f"{name}_localization"),
            getattr(args, f"{name}_motive"),
            front_plane_to_pelvis_x_m=args.front_plane_to_pelvis_x_m,
            motive_rigid_body_id=args.motive_rigid_body_id,
            motive_rigid_body_name=args.motive_rigid_body_name,
            treatments_already_map=args.treatments_already_map,
        )
        start = getattr(args, f"{name}_start_ns")
        admitted = (full["event_time_ns"] >= start) & (
            full["event_time_ns"] <= start + round(args.fit_duration_sec * 1e9)
        )
        full_runs[name] = full
        fit_runs[name] = {
            key: value[admitted] if isinstance(value, np.ndarray) else value
            for key, value in full.items()
        }
    fitted = {name: _fit_transform(run) for name, run in fit_runs.items()}
    angle = [math.atan2(value[0][0, 1], value[0][0, 0]) for value in fitted.values()]
    translation = [value[1] for value in fitted.values()]
    report = {
        "schema": "g1_polycam_cross_run_consistency_v1",
        "motive_role": "evaluator_only",
        "motive_online_input": False,
        "front_plane_to_pelvis_x_m": args.front_plane_to_pelvis_x_m,
        "fit_duration_sec": args.fit_duration_sec,
        "initial_window": {
            "walk02_to_walk03": _cross_score(fit_runs["walk02"], fit_runs["walk03"]),
            "walk03_to_walk02": _cross_score(fit_runs["walk03"], fit_runs["walk02"]),
        },
        "full_run": {
            "walk02_to_walk03": _cross_score(fit_runs["walk02"], full_runs["walk03"]),
            "walk03_to_walk02": _cross_score(fit_runs["walk03"], full_runs["walk02"]),
        },
        "independent_motive_to_map_transform_delta": {
            "yaw_deg": float(math.degrees(_shortest(np.asarray(angle[1] - angle[0])))),
            "translation_m": float(np.linalg.norm(translation[1] - translation[0])),
        },
        "interpretation_contract": (
            "cross-run consistency can reject a false map placement but does not make "
            "the Polycam mesh or Motive-to-pelvis lever arm ground truth"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
