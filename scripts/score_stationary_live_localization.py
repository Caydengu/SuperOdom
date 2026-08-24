#!/usr/bin/env python3
"""Score a passive stationary G1 localization track against Motive pelvis truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from g1_root_state_bridge.clock_sync import fit_affine_lower_envelope_clock

FRONT_PLANE_TO_PELVIS_X_M = -0.061431244015693665


def distribution(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "rmse": float(np.sqrt(np.mean(finite**2))),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "maximum": float(np.max(finite)),
    }


def rotation_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm((x, y, z, w)))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid quaternion")
    x, y, z, w = (x / norm, y / norm, z / norm, w / norm)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def z_up_yaw(quaternion: np.ndarray) -> float:
    rotation = rotation_from_xyzw(quaternion)
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def motive_y_up_yaw(quaternion: np.ndarray) -> float:
    rotation = rotation_from_xyzw(quaternion)
    forward = rotation[:, 0]
    return math.atan2(float(-forward[2]), float(forward[0]))


def shortest(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def body_aligned(planar_delta: np.ndarray, initial_yaw: float) -> np.ndarray:
    c, s = math.cos(initial_yaw), math.sin(initial_yaw)
    world_from_body = np.asarray(((c, -s), (s, c)), dtype=np.float64)
    return planar_delta @ world_from_body


def load_odometry(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") == "odometry" and row.get("topic") == "/g1/localization/pelvis_odom":
                rows.append(row)
    if len(rows) < 3:
        raise ValueError("odometry track has fewer than three samples")
    event_ns = np.asarray([row["source_time_ns"] for row in rows], dtype=np.int64)
    receipt_ns = np.asarray([row["receipt_time_ns"] for row in rows], dtype=np.int64)
    if not np.all(np.diff(event_ns) > 0):
        raise ValueError("odometry source time is not strictly increasing")
    return {
        "event_ns": event_ns,
        "receipt_ns": receipt_ns,
        "position": np.asarray([row["position_xyz_m"] for row in rows], dtype=np.float64),
        "quaternion": np.asarray([row["quaternion_xyzw"] for row in rows], dtype=np.float64),
    }


def load_motive(path: Path, rigid_body_id: int, rigid_body_name: str) -> dict[str, np.ndarray]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("record_type") == "metadata":
                if int(row["requested_rigid_body_id"]) != rigid_body_id:
                    raise ValueError("Motive metadata has the wrong rigid-body ID")
                if row["rigid_body_name"] != rigid_body_name:
                    raise ValueError("Motive metadata has the wrong rigid-body name")
                continue
            if (
                row.get("record_type") == "frame"
                and row.get("tracking_valid")
                and int(row["rigid_body_id"]) == rigid_body_id
                and row["rigid_body_name"] == rigid_body_name
            ):
                rows.append(row)
    if len(rows) < 3:
        raise ValueError("Motive track has fewer than three valid frames")
    software_ns = np.asarray(
        [round(float(row["motive_software_time_s"]) * 1e9) for row in rows],
        dtype=np.int64,
    )
    receipt_ns = np.asarray([row["receipt_realtime_ns"] for row in rows], dtype=np.int64)
    clock = fit_affine_lower_envelope_clock(software_ns, receipt_ns, lower_quantile=0.01)
    event_ns = np.asarray([clock.map_time_ns(int(value)) for value in software_ns], dtype=np.int64)
    quaternion = np.asarray(
        [row["quaternion_xyzw_motive_native"] for row in rows],
        dtype=np.float64,
    )
    marker = np.asarray(
        [row["position_xyz_m_motive_native"] for row in rows],
        dtype=np.float64,
    )
    offset = np.asarray((FRONT_PLANE_TO_PELVIS_X_M, 0.0, 0.0), dtype=np.float64)
    pelvis = np.asarray(
        [position + rotation_from_xyzw(q) @ offset for position, q in zip(marker, quaternion, strict=True)],
        dtype=np.float64,
    )
    return {
        "event_ns": event_ns,
        "position": pelvis,
        "quaternion": quaternion,
        "clock_residual_p95_ns": np.asarray((clock.residual_p95_ns,)),
    }


def score(
    odometry: dict[str, np.ndarray],
    motive: dict[str, np.ndarray],
    *,
    warmup_sec: float,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    start_ns = max(int(odometry["event_ns"][0]), int(motive["event_ns"][0])) + round(warmup_sec * 1e9)
    end_ns = min(int(odometry["event_ns"][-1]), int(motive["event_ns"][-1]))
    admitted = (odometry["event_ns"] >= start_ns) & (odometry["event_ns"] <= end_ns)
    event_ns = odometry["event_ns"][admitted]
    if event_ns.size < 100 or end_ns <= start_ns:
        raise ValueError("stationary scoring overlap is too short")
    position = odometry["position"][admitted]
    quaternion = odometry["quaternion"][admitted]
    seconds = (motive["event_ns"] - motive["event_ns"][0]).astype(np.float64) * 1e-9
    query = (event_ns - motive["event_ns"][0]).astype(np.float64) * 1e-9
    motive_position = np.column_stack(
        [np.interp(query, seconds, motive["position"][:, axis]) for axis in range(3)]
    )
    estimate_yaw = np.unwrap(np.asarray([z_up_yaw(q) for q in quaternion]))
    motive_yaw_all = np.unwrap(np.asarray([motive_y_up_yaw(q) for q in motive["quaternion"]]))
    motive_yaw = np.interp(query, seconds, motive_yaw_all)

    estimate_delta = position - position[0]
    motive_delta = motive_position - motive_position[0]
    estimate_body = body_aligned(estimate_delta[:, :2], float(estimate_yaw[0]))
    motive_planar = np.column_stack((motive_delta[:, 0], -motive_delta[:, 2]))
    motive_body = body_aligned(motive_planar, float(motive_yaw[0]))
    estimate_relative_yaw = estimate_yaw - estimate_yaw[0]
    motive_relative_yaw = motive_yaw - motive_yaw[0]
    planar_error = np.linalg.norm(estimate_body - motive_body, axis=1)
    yaw_error_deg = np.degrees(np.abs(shortest(estimate_relative_yaw - motive_relative_yaw)))
    estimate_excursion = np.linalg.norm(estimate_body, axis=1)
    motive_excursion = np.linalg.norm(motive_body, axis=1)
    pose_age_ms = (
        odometry["receipt_ns"][admitted] - odometry["event_ns"][admitted]
    ).astype(np.float64) * 1e-6
    gap_ms = np.diff(event_ns).astype(np.float64) * 1e-6
    jump_m = np.linalg.norm(np.diff(estimate_body, axis=0), axis=1)
    duration_sec = float((event_ns[-1] - event_ns[0]) * 1e-9)
    terminal_planar_drift_m = float(estimate_excursion[-1])
    terminal_yaw_drift_deg = float(abs(math.degrees(estimate_relative_yaw[-1])))
    passed = (
        terminal_planar_drift_m <= 0.03
        and terminal_yaw_drift_deg <= 2.0
        and float(np.max(jump_m)) <= 0.25
        and float(np.quantile(pose_age_ms, 0.95)) <= 50.0
    )
    metrics: dict[str, Any] = {
        "schema": "g1_kiss_stationary_motive_score_v1",
        "status": "pass" if passed else "fail",
        "reference_role": "evaluator_only",
        "alignment": "initial pelvis heading only; no trajectory fit and no scale fit",
        "front_plane_to_pelvis_x_m": FRONT_PLANE_TO_PELVIS_X_M,
        "duration_sec": duration_sec,
        "sample_count": int(event_ns.size),
        "nominal_rate_hz": float(1e3 / np.median(gap_ms)),
        "terminal_planar_drift_m": terminal_planar_drift_m,
        "terminal_yaw_drift_deg": terminal_yaw_drift_deg,
        "estimate_planar_excursion_m": distribution(estimate_excursion),
        "motive_planar_excursion_m": distribution(motive_excursion),
        "planar_error_m": distribution(planar_error),
        "yaw_error_deg": distribution(yaw_error_deg),
        "pose_age_ms": distribution(pose_age_ms),
        "source_gap_ms": distribution(gap_ms),
        "maximum_consecutive_planar_step_m": float(np.max(jump_m)),
        "motive_clock_residual_p95_ms": float(motive["clock_residual_p95_ns"][0] * 1e-6),
        "thresholds": {
            "terminal_planar_drift_m": 0.03,
            "terminal_yaw_drift_deg": 2.0,
            "maximum_consecutive_planar_step_m": 0.25,
            "pose_age_p95_ms": 50.0,
        },
    }
    samples = [
        {
            "time_s": float((event_ns[index] - event_ns[0]) * 1e-9),
            "estimate_x_m": float(estimate_body[index, 0]),
            "estimate_y_m": float(estimate_body[index, 1]),
            "motive_x_m": float(motive_body[index, 0]),
            "motive_y_m": float(motive_body[index, 1]),
            "planar_error_m": float(planar_error[index]),
            "estimate_yaw_deg": float(math.degrees(estimate_relative_yaw[index])),
            "motive_yaw_deg": float(math.degrees(motive_relative_yaw[index])),
            "yaw_error_deg": float(yaw_error_deg[index]),
            "pose_age_ms": float(pose_age_ms[index]),
        }
        for index in range(event_ns.size)
    ]
    return metrics, samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odometry", type=Path, required=True)
    parser.add_argument("--motive", type=Path, required=True)
    parser.add_argument("--rigid-body-id", type=int, default=42)
    parser.add_argument("--rigid-body-name", default="G1_PELVIS_F_4123")
    parser.add_argument("--warmup-sec", type=float, default=5.0)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.metrics, args.samples):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    metrics, samples = score(
        load_odometry(args.odometry),
        load_motive(args.motive, args.rigid_body_id, args.rigid_body_name),
        warmup_sec=args.warmup_sec,
    )
    args.metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.samples.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
