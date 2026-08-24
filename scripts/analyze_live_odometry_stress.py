#!/usr/bin/env python3
"""Analyze a live local-odometry capture by Motive-derived motion regime.

Motive is used only after the run as an evaluator.  The estimator trajectory is
aligned once over an explicit initial fit interval; no scale is fitted and no
per-segment reset or alignment is allowed.  Motion labels are derived only from
the reference kinematics, never from localization error.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from g1_root_state_bridge.amo_dataset import load_motive
from g1_root_state_bridge.amo_scoring import (
    _distribution,
    _interpolate_reference,
    _robotics_yaw_z_up,
    _rpe,
    _shortest,
    provisional_reference,
)

from scripts.score_initial_fit_local_odometry import (
    _fit_rigid_row,
    _load_odometry,
)

MOTION_THRESHOLDS = {
    "smoothing_window_s": 0.5,
    "stationary_speed_mps": 0.08,
    "stationary_yaw_rate_degps": 8.0,
    "turning_yaw_rate_degps": 15.0,
    "pure_turn_speed_mps": 0.12,
    "axis_dominance_ratio": 1.25,
}


def _smooth(values: np.ndarray, sample_count: int) -> np.ndarray:
    count = max(1, int(sample_count))
    if count == 1:
        return values.copy()
    if count % 2 == 0:
        count += 1
    radius = count // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.full(count, 1.0 / count, dtype=np.float64)
    return np.column_stack(
        [
            np.convolve(padded[:, axis], kernel, mode="valid")
            for axis in range(values.shape[1])
        ]
    )


def _smooth_1d(values: np.ndarray, sample_count: int) -> np.ndarray:
    return _smooth(np.asarray(values, dtype=np.float64)[:, None], sample_count)[:, 0]


def _motion_kinematics(
    time_ns: np.ndarray,
    reference_xy: np.ndarray,
    reference_yaw: np.ndarray,
) -> dict[str, np.ndarray]:
    relative_s = (time_ns - time_ns[0]).astype(np.float64) * 1e-9
    nominal_rate_hz = 1.0 / float(np.median(np.diff(relative_s)))
    smooth_count = max(
        3, round(MOTION_THRESHOLDS["smoothing_window_s"] * nominal_rate_hz)
    )
    xy = _smooth(reference_xy, smooth_count)
    yaw = _smooth_1d(np.unwrap(reference_yaw), smooth_count)
    velocity = np.column_stack(
        [np.gradient(xy[:, axis], relative_s) for axis in range(2)]
    )
    speed = np.linalg.norm(velocity, axis=1)
    yaw_rate = np.gradient(yaw, relative_s)
    acceleration = np.gradient(speed, relative_s)
    forward = np.column_stack((np.cos(yaw), np.sin(yaw)))
    left = np.column_stack((-np.sin(yaw), np.cos(yaw)))
    forward_speed = np.sum(velocity * forward, axis=1)
    lateral_speed = np.sum(velocity * left, axis=1)
    return {
        "speed_mps": speed,
        "yaw_rate_radps": yaw_rate,
        "acceleration_mps2": acceleration,
        "forward_speed_mps": forward_speed,
        "lateral_speed_mps": lateral_speed,
    }


def classify_motion(kinematics: dict[str, np.ndarray]) -> np.ndarray:
    """Assign reference-only, order-independent motion labels."""

    speed = kinematics["speed_mps"]
    yaw_rate_degps = np.abs(np.degrees(kinematics["yaw_rate_radps"]))
    forward = np.abs(kinematics["forward_speed_mps"])
    lateral = np.abs(kinematics["lateral_speed_mps"])
    labels = np.full(speed.shape, "transition", dtype="U24")

    stationary = (speed < MOTION_THRESHOLDS["stationary_speed_mps"]) & (
        yaw_rate_degps < MOTION_THRESHOLDS["stationary_yaw_rate_degps"]
    )
    pure_turn = (speed < MOTION_THRESHOLDS["pure_turn_speed_mps"]) & (
        yaw_rate_degps >= MOTION_THRESHOLDS["turning_yaw_rate_degps"]
    )
    combined = (speed >= MOTION_THRESHOLDS["stationary_speed_mps"]) & (
        yaw_rate_degps >= MOTION_THRESHOLDS["turning_yaw_rate_degps"]
    )
    low_turn = yaw_rate_degps < MOTION_THRESHOLDS["turning_yaw_rate_degps"]
    longitudinal = (
        (speed >= MOTION_THRESHOLDS["stationary_speed_mps"])
        & low_turn
        & (forward >= MOTION_THRESHOLDS["axis_dominance_ratio"] * lateral)
    )
    lateral_motion = (
        (speed >= MOTION_THRESHOLDS["stationary_speed_mps"])
        & low_turn
        & (lateral >= MOTION_THRESHOLDS["axis_dominance_ratio"] * forward)
    )
    diagonal = (
        (speed >= MOTION_THRESHOLDS["stationary_speed_mps"])
        & low_turn
        & ~longitudinal
        & ~lateral_motion
    )
    labels[stationary] = "stationary"
    labels[pure_turn] = "pure_turn"
    labels[combined] = "translation_and_turn"
    labels[longitudinal] = "longitudinal"
    labels[lateral_motion] = "lateral"
    labels[diagonal] = "diagonal"
    return labels


def _masked_rpe(
    time_ns: np.ndarray,
    predicted_xy: np.ndarray,
    reference_xy: np.ndarray,
    predicted_yaw: np.ndarray,
    reference_yaw: np.ndarray,
    mask: np.ndarray,
    horizon_s: float,
) -> dict[str, object]:
    target = time_ns + round(horizon_s * 1e9)
    right = np.searchsorted(time_ns, target, side="left")
    valid = (right < time_ns.size) & mask
    left = np.flatnonzero(valid)
    right = right[valid]
    timing_error_ms = np.abs(time_ns[right] - target[valid]) * 1e-6
    keep = timing_error_ms <= 15.0
    left, right, timing_error_ms = left[keep], right[keep], timing_error_ms[keep]
    if not left.size:
        return {
            "horizon_s": horizon_s,
            "timing_error_ms": {"count": 0},
            "translation_error_m": {"count": 0},
            "yaw_error_deg": {"count": 0},
        }
    translation_error = np.linalg.norm(
        (predicted_xy[right] - predicted_xy[left])
        - (reference_xy[right] - reference_xy[left]),
        axis=1,
    )
    yaw_error = np.abs(
        _shortest(
            (predicted_yaw[right] - predicted_yaw[left])
            - (reference_yaw[right] - reference_yaw[left])
        )
    )
    return {
        "horizon_s": horizon_s,
        "timing_error_ms": _distribution(timing_error_ms),
        "translation_error_m": _distribution(translation_error),
        "yaw_error_deg": _distribution(np.degrees(yaw_error)),
    }


def analyze_live_odometry(
    *,
    odometry_path: Path,
    motive_path: Path,
    start_source_ns: int,
    fit_duration_sec: float,
    front_plane_to_pelvis_x_m: float,
    motive_rigid_body_id: int,
    motive_rigid_body_name: str,
    window_sec: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    records = _load_odometry(odometry_path, "/g1/localization/pelvis_odom")
    time_ns = np.asarray([row["source_time_ns"] for row in records], dtype=np.int64)
    position = np.asarray([row["position_xyz_m"] for row in records], dtype=np.float64)
    quaternion = np.asarray(
        [row["quaternion_xyzw"] for row in records], dtype=np.float64
    )
    motive = load_motive(
        motive_path,
        expected_rigid_body_id=motive_rigid_body_id,
        expected_rigid_body_name=motive_rigid_body_name,
    )
    ref_time, ref_position, ref_yaw = provisional_reference(
        motive, front_plane_to_pelvis_x_m=front_plane_to_pelvis_x_m
    )
    admitted = (
        (time_ns >= start_source_ns)
        & (time_ns >= ref_time[0])
        & (time_ns <= ref_time[-1])
    )
    time_ns = time_ns[admitted]
    position = position[admitted]
    quaternion = quaternion[admitted]
    if time_ns.size < 10:
        raise ValueError("fewer than ten live odometry samples overlap Motive")
    ref_position, ref_yaw = _interpolate_reference(
        ref_time, ref_position, ref_yaw, time_ns
    )
    reference_xy = np.column_stack((ref_position[:, 0], -ref_position[:, 2]))
    local_xy = position[:, :2]
    fit = time_ns <= start_source_ns + round(fit_duration_sec * 1e9)
    if np.sum(fit) < 10:
        raise ValueError("fewer than ten samples in the initial fit interval")
    rotation, translation = _fit_rigid_row(local_xy[fit], reference_xy[fit])
    predicted_xy = local_xy @ rotation + translation
    local_yaw = _robotics_yaw_z_up(quaternion)
    yaw_offset = float(
        np.arctan2(
            np.mean(np.sin(ref_yaw[fit] - local_yaw[fit])),
            np.mean(np.cos(ref_yaw[fit] - local_yaw[fit])),
        )
    )
    predicted_yaw = local_yaw + yaw_offset
    planar_error = np.linalg.norm(predicted_xy - reference_xy, axis=1)
    signed_yaw_error_deg = np.degrees(_shortest(predicted_yaw - ref_yaw))
    kinematics = _motion_kinematics(time_ns, reference_xy, ref_yaw)
    labels = classify_motion(kinematics)
    nominal_rate_hz = 1.0 / float(np.median(np.diff(time_ns).astype(np.float64) * 1e-9))

    by_regime: dict[str, object] = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        by_regime[label] = {
            "sample_count": int(np.sum(mask)),
            "approximate_duration_s": float(np.sum(mask) / nominal_rate_hz),
            "reference_speed_mps": _distribution(kinematics["speed_mps"][mask]),
            "reference_abs_yaw_rate_degps": _distribution(
                np.abs(np.degrees(kinematics["yaw_rate_radps"][mask]))
            ),
            "planar_error_m": _distribution(planar_error[mask]),
            "yaw_error_deg": _distribution(np.abs(signed_yaw_error_deg[mask])),
            "rpe": {
                str(horizon): _masked_rpe(
                    time_ns,
                    predicted_xy,
                    reference_xy,
                    predicted_yaw,
                    ref_yaw,
                    mask,
                    horizon,
                )
                for horizon in (1.0, 5.0, 10.0)
            },
        }

    windows: list[dict[str, object]] = []
    window_ns = round(window_sec * 1e9)
    for window_index, begin_ns in enumerate(
        range(int(time_ns[0]), int(time_ns[-1]) + 1, window_ns)
    ):
        mask = (time_ns >= begin_ns) & (time_ns < begin_ns + window_ns)
        if np.sum(mask) < 3:
            continue
        values, counts = np.unique(labels[mask], return_counts=True)
        dominant = str(values[int(np.argmax(counts))])
        windows.append(
            {
                "window_index": window_index,
                "start_s": float((begin_ns - time_ns[0]) * 1e-9),
                "end_s": float(
                    (min(begin_ns + window_ns, int(time_ns[-1])) - time_ns[0]) * 1e-9
                ),
                "sample_count": int(np.sum(mask)),
                "dominant_regime": dominant,
                "regime_fractions": {
                    str(value): float(count / np.sum(mask))
                    for value, count in zip(values, counts)
                },
                "planar_error_m": _distribution(planar_error[mask]),
                "yaw_error_deg": _distribution(np.abs(signed_yaw_error_deg[mask])),
                "reference_speed_mps": _distribution(kinematics["speed_mps"][mask]),
                "reference_abs_yaw_rate_degps": _distribution(
                    np.abs(np.degrees(kinematics["yaw_rate_radps"][mask]))
                ),
            }
        )

    rows = []
    for index in range(time_ns.size):
        rows.append(
            {
                "schema": "g1_live_odometry_stress_sample_v1",
                "source_time_ns": int(time_ns[index]),
                "elapsed_s": float((time_ns[index] - time_ns[0]) * 1e-9),
                "regime": str(labels[index]),
                "reference_xy_m": reference_xy[index].tolist(),
                "predicted_xy_m": predicted_xy[index].tolist(),
                "planar_error_m": float(planar_error[index]),
                "reference_yaw_deg": float(math.degrees(ref_yaw[index])),
                "predicted_yaw_deg": float(math.degrees(predicted_yaw[index])),
                "signed_yaw_error_deg": float(signed_yaw_error_deg[index]),
                "reference_speed_mps": float(kinematics["speed_mps"][index]),
                "reference_yaw_rate_degps": float(
                    math.degrees(kinematics["yaw_rate_radps"][index])
                ),
                "reference_forward_speed_mps": float(
                    kinematics["forward_speed_mps"][index]
                ),
                "reference_lateral_speed_mps": float(
                    kinematics["lateral_speed_mps"][index]
                ),
                "reference_acceleration_mps2": float(
                    kinematics["acceleration_mps2"][index]
                ),
            }
        )

    report = {
        "schema": "g1_live_odometry_stress_analysis_v1",
        "reference_role": "evaluator_only",
        "motive_online_input": False,
        "motion_labels_use_localization_error": False,
        "motion_order_required": False,
        "alignment": {
            "method": "one initial evaluator-only SE2 fit; no scale",
            "start_source_ns": start_source_ns,
            "fit_duration_sec": fit_duration_sec,
            "rotation_deg": math.degrees(math.atan2(rotation[0, 1], rotation[0, 0])),
            "translation_m": translation.tolist(),
            "yaw_offset_deg": math.degrees(yaw_offset),
        },
        "motion_thresholds": MOTION_THRESHOLDS,
        "sample_count": int(time_ns.size),
        "duration_s": float((time_ns[-1] - time_ns[0]) * 1e-9),
        "nominal_rate_hz": nominal_rate_hz,
        "reference_path_length_m": float(
            np.sum(np.linalg.norm(np.diff(reference_xy, axis=0), axis=1))
        ),
        "overall": {
            "planar_error_m": _distribution(planar_error),
            "yaw_error_deg": _distribution(np.abs(signed_yaw_error_deg)),
            "rpe": {
                str(horizon): _rpe(
                    time_ns,
                    predicted_xy,
                    reference_xy,
                    predicted_yaw,
                    ref_yaw,
                    horizon,
                )
                for horizon in (1.0, 5.0, 10.0)
            },
            "final_planar_error_m": float(planar_error[-1]),
            "final_signed_yaw_error_deg": float(signed_yaw_error_deg[-1]),
        },
        "by_motion_regime": by_regime,
        "fixed_windows": windows,
        "worst_windows": {
            "planar_rmse": sorted(
                windows, key=lambda row: row["planar_error_m"]["rmse"], reverse=True
            )[:5],
            "yaw_rmse": sorted(
                windows, key=lambda row: row["yaw_error_deg"]["rmse"], reverse=True
            )[:5],
        },
    }
    return report, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odometry", type=Path, required=True)
    parser.add_argument("--motive", type=Path, required=True)
    parser.add_argument("--start-source-ns", type=int, required=True)
    parser.add_argument("--fit-duration-sec", type=float, default=1.5)
    parser.add_argument("--front-plane-to-pelvis-x-m", type=float, default=0.0)
    parser.add_argument("--motive-rigid-body-id", type=int, default=42)
    parser.add_argument("--motive-rigid-body-name", default="G1_PELVIS_F_4123")
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.samples_output.exists():
        raise FileExistsError("refusing to overwrite stress-analysis output")
    report, rows = analyze_live_odometry(
        odometry_path=args.odometry,
        motive_path=args.motive,
        start_source_ns=args.start_source_ns,
        fit_duration_sec=args.fit_duration_sec,
        front_plane_to_pelvis_x_m=args.front_plane_to_pelvis_x_m,
        motive_rigid_body_id=args.motive_rigid_body_id,
        motive_rigid_body_name=args.motive_rigid_body_name,
        window_sec=args.window_sec,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.samples_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.samples_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples_output": str(args.samples_output),
                "sample_count": len(rows),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
