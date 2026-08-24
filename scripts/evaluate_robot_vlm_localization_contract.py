#!/usr/bin/env python3
"""Replay the exact robot-vlm two-lane contract and score it cross-run.

Motive is loaded only after both onboard lanes have been composed.  A rigid
Motive-to-map transform is fitted on the other run's initial window, so the
reported map errors measure cross-run consistency rather than leaking Motive
into localization.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

from g1_root_state_bridge.amo_dataset import load_motive
from g1_root_state_bridge.amo_scoring import (
    _interpolate_reference,
    provisional_reference,
)
from g1_root_state_bridge.amo_treatments import rotation_from_xyzw


def _shortest(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _load_records(path: Path, treatment: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            if payload.get("treatment") == treatment:
                records.append(payload)
    if len(records) < 2:
        raise ValueError(f"missing treatment {treatment!r} in {path}")
    return records


def _pose(row: dict[str, object]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation_from_xyzw(
        np.asarray(row["quaternion_xyzw"], dtype=np.float64)
    )
    result[:3, 3] = np.asarray(row["position_xyz_m"], dtype=np.float64)
    return result


def _map_transform(row: dict[str, object]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    # Offline registration stores row-vector XY transforms.  RVMAP001 carries
    # the equivalent column-vector homogeneous transform.
    rotation_row = np.asarray(row["rotation_matrix"], dtype=np.float64)
    result[:2, :2] = rotation_row.T
    result[:2, 3] = np.asarray(row["translation_m"], dtype=np.float64)
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


def _circular_mean(angle: np.ndarray) -> float:
    return float(np.arctan2(np.mean(np.sin(angle)), np.mean(np.cos(angle))))


def _run_contract(
    *,
    records: list[dict[str, object]],
    correction_report: Path,
    localization_module: object,
    position_policy: str,
    fit_start_ns: int,
    fit_duration_sec: float,
) -> dict[str, np.ndarray | dict[str, object]]:
    LocalHealth = localization_module.LocalHealth
    LocalOdometrySample = localization_module.LocalOdometrySample
    MapCorrectionCandidate = localization_module.MapCorrectionCandidate
    MapHealth = localization_module.MapHealth
    state = localization_module.LayeredLocalizationState(
        map_digest=b"p" * 32,
        require_root_imu_fusion=True,
        map_config=localization_module.MapCorrectionConfig(
            position_policy=position_policy
        ),
    )
    report = json.loads(correction_report.read_text(encoding="utf-8"))
    correction_rows = [row for row in report["corrections"] if row["accepted"]]
    initial_transform = _map_transform(correction_rows[0])
    fit_rows = [
        row
        for row in records
        if fit_start_ns
        <= int(row["source_time_ns"])
        <= fit_start_ns + round(fit_duration_sec * 1e9)
    ]
    if len(fit_rows) < 10:
        raise ValueError("fewer than ten local samples in the map-initialization window")
    fit_poses = np.stack([initial_transform @ _pose(row) for row in fit_rows])
    correction_index = 0
    map_sequence = 0
    output_time: list[int] = []
    output_pose: list[np.ndarray] = []
    output_quality: list[float] = []
    output_status: list[str] = []
    output_map_age_s: list[float] = []
    output_local_age_s: list[float] = []
    accepted_map_updates = 0
    rejected_by_runtime = 0
    for sequence, row in enumerate(records, start=1):
        source_time_ns = int(row["source_time_ns"])
        runtime_ns = max(1_000_000, round(float(row.get("runtime_ms", 8.0)) * 1e6))
        receipt_time_ns = source_time_ns + runtime_ns + 2_000_000
        flags = localization_module.REQUIRED_FUSED_LOCAL_HEALTH
        if not bool(row.get("deskew_valid", True)):
            flags &= ~LocalHealth.INERTIAL_DESKEW_VALID
        if not bool(row.get("orientation_fusion_healthy", True)):
            flags &= ~LocalHealth.HEADING_VALID
        local = LocalOdometrySample(
            sequence=sequence,
            source_epoch=1,
            estimate_time_ns=source_time_ns,
            publish_time_ns=receipt_time_ns - 1_000_000,
            receipt_time_ns=receipt_time_ns,
            correction_time_ns=source_time_ns,
            health_flags=LocalHealth(flags),
            local_T_pelvis=_pose(row),
            linear_velocity_world=(0.0, 0.0, 0.0),
            angular_velocity_world=(0.0, 0.0, 0.0),
            covariance_diagonal=(0.01,) * 6,
            calibration_digest=b"c" * 32,
        )
        state.update_local(local)
        while (
            correction_index < len(correction_rows)
            and int(correction_rows[correction_index]["available_after_source_ns"])
            <= source_time_ns
        ):
            correction = correction_rows[correction_index]
            correction_index += 1
            map_sequence += 1
            evidence_time_ns = int(correction["available_after_source_ns"])
            quality = correction["quality"]
            observability = quality["observability"]
            candidate = MapCorrectionCandidate(
                sequence=map_sequence,
                map_epoch=1,
                local_source_epoch=1,
                map_version=1,
                reference_time_ns=max(0, evidence_time_ns - 5_000_000_000),
                evidence_time_ns=evidence_time_ns,
                application_time_ns=evidence_time_ns,
                publish_time_ns=evidence_time_ns + 1_000_000,
                receipt_time_ns=evidence_time_ns + 2_000_000,
                health_flags=MapHealth(localization_module.REQUIRED_MAP_HEALTH),
                map_T_local=_map_transform(correction),
                fitness=float(quality["inlier_fraction"]),
                rmse_m=float(quality["rmse_m"]),
                min_eig=float(observability["minimum_eigenvalue"]),
                cond_number=float(observability["condition_number"]),
                covariance_diagonal=(0.01,) * 6,
                map_digest=b"p" * 32,
            )
            decision = state.consider_map(candidate)
            if decision.accepted:
                accepted_map_updates += 1
            else:
                rejected_by_runtime += 1
        health = state.health(now_ns=receipt_time_ns)
        if "map_uninitialized" in health.reasons:
            continue
        output_time.append(source_time_ns)
        output_pose.append(state.map_pose())
        output_quality.append(health.quality)
        output_status.append(health.status)
        output_map_age_s.append(health.map_age_s)
        output_local_age_s.append(health.local_age_s)
    if not output_time:
        raise ValueError("no map-initialized contract output")
    poses = np.stack(output_pose)
    return {
        "time_ns": np.asarray(output_time, dtype=np.int64),
        "position_xy": poses[:, :2, 3],
        "yaw_rad": np.arctan2(poses[:, 1, 0], poses[:, 0, 0]),
        "quality": np.asarray(output_quality, dtype=np.float64),
        "status": np.asarray(output_status),
        "map_age_s": np.asarray(output_map_age_s, dtype=np.float64),
        "local_age_s": np.asarray(output_local_age_s, dtype=np.float64),
        "fit_time_ns": np.asarray(
            [row["source_time_ns"] for row in fit_rows], dtype=np.int64
        ),
        "fit_position_xy": fit_poses[:, :2, 3],
        "fit_yaw_rad": np.arctan2(fit_poses[:, 1, 0], fit_poses[:, 0, 0]),
        "audit": {
            "input_record_count": len(records),
            "output_record_count": len(output_time),
            "publisher_accepted_map_update_count": len(correction_rows),
            "runtime_accepted_map_update_count": accepted_map_updates,
            "runtime_rejected_map_update_count": rejected_by_runtime,
        },
    }


def _attach_motive(
    run: dict[str, object],
    *,
    motive_path: Path,
    front_plane_to_pelvis_x_m: float,
    rigid_body_id: int,
    rigid_body_name: str,
) -> None:
    motive = load_motive(
        motive_path,
        expected_rigid_body_id=rigid_body_id,
        expected_rigid_body_name=rigid_body_name,
    )
    reference_time_ns, reference_position, reference_yaw = provisional_reference(
        motive,
        front_plane_to_pelvis_x_m=front_plane_to_pelvis_x_m,
    )
    query = np.asarray(run["time_ns"], dtype=np.int64)
    valid = (query >= reference_time_ns[0]) & (query <= reference_time_ns[-1])
    for key in (
        "time_ns",
        "position_xy",
        "yaw_rad",
        "quality",
        "status",
        "map_age_s",
        "local_age_s",
    ):
        run[key] = np.asarray(run[key])[valid]
    position, yaw = _interpolate_reference(
        reference_time_ns, reference_position, reference_yaw, query[valid]
    )
    run["motive_position_xy"] = np.column_stack((position[:, 0], -position[:, 2]))
    run["motive_yaw_rad"] = yaw
    fit_query = np.asarray(run["fit_time_ns"], dtype=np.int64)
    fit_valid = (fit_query >= reference_time_ns[0]) & (
        fit_query <= reference_time_ns[-1]
    )
    for key in ("fit_time_ns", "fit_position_xy", "fit_yaw_rad"):
        run[key] = np.asarray(run[key])[fit_valid]
    fit_position, fit_yaw = _interpolate_reference(
        reference_time_ns,
        reference_position,
        reference_yaw,
        fit_query[fit_valid],
    )
    run["fit_motive_position_xy"] = np.column_stack(
        (fit_position[:, 0], -fit_position[:, 2])
    )
    run["fit_motive_yaw_rad"] = fit_yaw


def _fit_motive_to_map(
    run: dict[str, object]
) -> tuple[np.ndarray, np.ndarray, float]:
    rotation, translation = _fit_rigid_row(
        np.asarray(run["fit_motive_position_xy"]),
        np.asarray(run["fit_position_xy"]),
    )
    yaw = _circular_mean(
        np.asarray(
            _shortest(
                np.asarray(run["fit_yaw_rad"])
                - np.asarray(run["fit_motive_yaw_rad"])
            )
        )
    )
    return rotation, translation, yaw


def _score(
    train: dict[str, object],
    test: dict[str, object],
    *,
    fit_duration_sec: float,
) -> dict[str, object]:
    rotation, translation, yaw_offset = _fit_motive_to_map(train)
    reference_xy = np.asarray(test["motive_position_xy"]) @ rotation + translation
    position_error = np.linalg.norm(reference_xy - np.asarray(test["position_xy"]), axis=1)
    yaw_error_deg = np.degrees(
        np.abs(
            _shortest(
                np.asarray(test["motive_yaw_rad"])
                + yaw_offset
                - np.asarray(test["yaw_rad"])
            )
        )
    )
    quality = np.asarray(test["quality"])
    healthy = quality >= 0.5
    severe_error = (position_error > 0.25) | (yaw_error_deg > 10.0)
    false_healthy = healthy & severe_error
    return {
        "sample_count": int(position_error.size),
        "position_rmse_m": float(np.sqrt(np.mean(position_error**2))),
        "position_p95_m": float(np.quantile(position_error, 0.95)),
        "yaw_rmse_deg": float(np.sqrt(np.mean(yaw_error_deg**2))),
        "yaw_p95_deg": float(np.quantile(yaw_error_deg, 0.95)),
        "availability_at_quality_ge_0_5": float(np.mean(healthy)),
        "false_healthy_fraction_all_samples": float(np.mean(false_healthy)),
        "false_healthy_fraction_among_healthy": float(
            np.sum(false_healthy) / max(1, np.sum(healthy))
        ),
        "severe_position_fraction": float(np.mean(position_error > 0.25)),
        "severe_yaw_fraction": float(np.mean(yaw_error_deg > 10.0)),
        "quality": {
            "minimum": float(np.min(quality)),
            "median": float(np.median(quality)),
            "p95": float(np.quantile(quality, 0.95)),
        },
        "pose_age_s": {
            "p95": float(np.quantile(test["local_age_s"], 0.95)),
            "maximum": float(np.max(test["local_age_s"])),
        },
        "heading_correction_age_s": {
            "p95": float(np.quantile(test["map_age_s"], 0.95)),
            "maximum": float(np.max(test["map_age_s"])),
        },
        "trained_motive_to_map": {
            "rotation_deg": math.degrees(math.atan2(rotation[0, 1], rotation[0, 0])),
            "translation_m": translation.tolist(),
            "heading_offset_deg": math.degrees(yaw_offset),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-vlm-src", type=Path, required=True)
    for run in ("walk02", "walk03"):
        parser.add_argument(f"--{run}-treatments", type=Path, required=True)
        parser.add_argument(f"--{run}-corrections", type=Path, required=True)
        parser.add_argument(f"--{run}-motive", type=Path, required=True)
        parser.add_argument(f"--{run}-fit-start-ns", type=int, required=True)
    parser.add_argument(
        "--treatment",
        help="Shared treatment name; overridden by per-run treatment arguments.",
    )
    parser.add_argument("--walk02-treatment")
    parser.add_argument("--walk03-treatment")
    parser.add_argument("--fit-duration-sec", type=float, default=10.0)
    parser.add_argument(
        "--position-policy",
        choices=("initialize_once_heading_only", "full_se3"),
        default="initialize_once_heading_only",
    )
    parser.add_argument("--front-plane-to-pelvis-x-m", type=float, default=0.0)
    parser.add_argument("--motive-rigid-body-id", type=int, default=42)
    parser.add_argument("--motive-rigid-body-name", default="G1_PELVIS_F_4123")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    sys.path.insert(0, str(args.robot_vlm_src))
    from rv.backends.sim_real import localization as localization_module

    runs: dict[str, dict[str, object]] = {}
    treatment_names: dict[str, str] = {}
    for name in ("walk02", "walk03"):
        treatment_name = getattr(args, f"{name}_treatment") or args.treatment
        if treatment_name is None:
            raise ValueError(
                f"{name} requires --treatment or --{name}-treatment"
            )
        treatment_names[name] = treatment_name
        run = _run_contract(
            records=_load_records(
                getattr(args, f"{name}_treatments"), treatment_name
            ),
            correction_report=getattr(args, f"{name}_corrections"),
            localization_module=localization_module,
            position_policy=args.position_policy,
            fit_start_ns=getattr(args, f"{name}_fit_start_ns"),
            fit_duration_sec=args.fit_duration_sec,
        )
        _attach_motive(
            run,
            motive_path=getattr(args, f"{name}_motive"),
            front_plane_to_pelvis_x_m=args.front_plane_to_pelvis_x_m,
            rigid_body_id=args.motive_rigid_body_id,
            rigid_body_name=args.motive_rigid_body_name,
        )
        runs[name] = run
    report = {
        "schema": "g1_robot_vlm_localization_contract_score_v1",
        "motive_role": "evaluator_only_cross_run_transform",
        "motive_online_input": False,
        "local_treatments": treatment_names,
        "map_composition": args.position_policy,
        "quality_threshold": 0.5,
        "severe_error_thresholds": {"position_m": 0.25, "yaw_deg": 10.0},
        "front_plane_to_pelvis_x_m": args.front_plane_to_pelvis_x_m,
        "fit_duration_sec": args.fit_duration_sec,
        "run_audit": {name: run["audit"] for name, run in runs.items()},
        "cross_run": {
            "walk02_to_walk03": _score(
                runs["walk02"], runs["walk03"], fit_duration_sec=args.fit_duration_sec
            ),
            "walk03_to_walk02": _score(
                runs["walk03"], runs["walk02"], fit_duration_sec=args.fit_duration_sec
            ),
        },
        "within_run_initial_fit": {
            "walk02": _score(
                runs["walk02"], runs["walk02"], fit_duration_sec=args.fit_duration_sec
            ),
            "walk03": _score(
                runs["walk03"], runs["walk03"], fit_duration_sec=args.fit_duration_sec
            ),
        },
        "evidence_boundary": (
            "Motive-to-map is cross-run fitted because neither the Polycam mesh nor "
            "the front-plane rigid-body lever arm is surveyed ground truth"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
