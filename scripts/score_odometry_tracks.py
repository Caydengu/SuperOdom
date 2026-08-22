#!/usr/bin/env python3
"""Compare sensor- and pelvis-frame odometry tracks against Motive pelvis data.

Absolute Motive-to-SuperOdometry calibration was not measured in the archived
trials, so the scorer uses one constant planar rotation per odometry topic and
reports relative trajectory errors.  It never treats a fitted time lag or
frame alignment as a calibrated deployment transform.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TOPICS = ("/state_estimation", "/pelvis_state_estimation")


def percentile(values: np.ndarray, q: float) -> float | None:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None
    return float(np.percentile(finite, q))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_motive(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows = load_jsonl(path)
    pose_rows = [row for row in rows if row.get("kind") == "raw_rigid_body_pose"]
    usable = [
        row
        for row in pose_rows
        if row.get("tracking_valid")
        and row.get("geometry_finite")
        and row.get("capture_realtime_estimate_ns") is not None
    ]
    return usable, len(pose_rows)


def load_tracks(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows = load_jsonl(path)
    return {
        topic: [
            row
            for row in rows
            if row.get("kind") == "odometry" and row.get("topic") == topic
        ]
        for topic in TOPICS
    }


def unwrap(values: np.ndarray) -> np.ndarray:
    return np.unwrap(values)


def yaw_z_up(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion_xyzw.T
    return unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_y_up(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion_xyzw.T
    return unwrap(np.arctan2(2.0 * (x * z + w * y), 1.0 - 2.0 * (x * x + y * y)))


def interp_columns(query_t: np.ndarray, source_t: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.interp(query_t, source_t, values[:, axis]) for axis in range(values.shape[1])]
    )


def optimal_planar_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(source.T @ target)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    return {
        "count": int(len(values)),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "maximum": float(np.max(values)) if len(values) else None,
        "rmse": float(np.sqrt(np.mean(values**2))) if len(values) else None,
    }


def score_track(
    *,
    query_t: np.ndarray,
    motive_planar: np.ndarray,
    motive_yaw: np.ndarray,
    track_t: np.ndarray,
    track_position: np.ndarray,
    track_quaternion: np.ndarray,
    rpe_horizon_s: float = 1.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    track_planar = interp_columns(query_t, track_t, track_position[:, :2])
    track_yaw = np.interp(query_t, track_t, yaw_z_up(track_quaternion))

    relative_motive = motive_planar - motive_planar[0]
    relative_track = track_planar - track_planar[0]
    rotation = optimal_planar_rotation(relative_track, relative_motive)
    aligned_track = relative_track @ rotation
    position_error = np.linalg.norm(aligned_track - relative_motive, axis=1)

    sample_dt = float(np.median(np.diff(query_t)))
    horizon_samples = max(1, int(round(rpe_horizon_s / sample_dt)))
    motive_delta = relative_motive[horizon_samples:] - relative_motive[:-horizon_samples]
    track_delta = aligned_track[horizon_samples:] - aligned_track[:-horizon_samples]
    rpe = np.linalg.norm(track_delta - motive_delta, axis=1)

    motive_step = np.diff(relative_motive, axis=0)
    track_step = np.diff(aligned_track, axis=0)
    motive_speed = np.linalg.norm(motive_step, axis=1) / sample_dt
    track_speed = np.linalg.norm(track_step, axis=1) / sample_dt
    speed_error = track_speed - motive_speed

    relative_motive_yaw = motive_yaw - motive_yaw[0]
    relative_track_yaw = track_yaw - track_yaw[0]
    yaw_candidates = {
        1: relative_track_yaw - relative_motive_yaw,
        -1: -relative_track_yaw - relative_motive_yaw,
    }
    yaw_sign = min(yaw_candidates, key=lambda sign: np.mean(yaw_candidates[sign] ** 2))
    yaw_error = yaw_candidates[yaw_sign]

    source_gaps_ms = np.diff(track_t) * 1000.0
    metrics = {
        "count": int(len(track_t)),
        "source_gap_ms": distribution(source_gaps_ms),
        "constant_planar_rotation": rotation.tolist(),
        "planar_position_error_m": distribution(position_error),
        "relative_position_error_1s_m": distribution(rpe),
        "speed_error_mps": {
            **distribution(np.abs(speed_error)),
            "bias": float(np.mean(speed_error)),
            "correlation": float(np.corrcoef(track_speed, motive_speed)[0, 1]),
        },
        "relative_yaw_error_deg": distribution(np.degrees(np.abs(yaw_error))),
        "motive_to_track_yaw_sign": yaw_sign,
        "terminal_relative_displacement_error_m": float(position_error[-1]),
        "maximum_consecutive_planar_step_m": float(
            np.max(np.linalg.norm(np.diff(track_planar, axis=0), axis=1))
        ),
    }
    samples = {
        "aligned_planar": aligned_track,
        "position_error": position_error,
        "relative_yaw": yaw_sign * relative_track_yaw,
        "yaw_error_deg": np.degrees(yaw_error),
    }
    return metrics, samples


def event_metrics(path: Path, start_ns: int, end_ns: int) -> dict[str, Any]:
    counts: dict[str, int] = {}
    packet_epochs: set[int] = set()
    packet_times: list[int] = []
    drop_times: list[int] = []
    for row in load_jsonl(path):
        event = str(row.get("event", "<none>"))
        event_time = row.get("estimate_time_ns", row.get("receipt_time_ns"))
        if event_time is None or not start_ns <= int(event_time) <= end_ns:
            continue
        counts[event] = counts.get(event, 0) + 1
        if event == "packet_published":
            packet_epochs.add(int(row["source_epoch"]))
            packet_times.append(int(row["estimate_time_ns"]))
        elif event in {"estimate_dropped", "joint_rejected"}:
            drop_times.append(int(event_time))
    gaps_ms = np.diff(np.asarray(packet_times, dtype=np.int64)) / 1e6
    return {
        "event_counts": dict(sorted(counts.items())),
        "source_epoch_count": len(packet_epochs),
        "packet_gap_ms": distribution(gaps_ms),
        "joint_or_estimate_failure_count": len(drop_times),
    }


def relative_improvement(sensor: float | None, pelvis: float | None) -> float | None:
    if sensor is None or pelvis is None or sensor == 0.0:
        return None
    return 100.0 * (sensor - pelvis) / sensor


def analyze_trial(
    *, motive_path: Path, tracks_path: Path, bridge_path: Path
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    motive_rows, raw_motive_count = load_motive(motive_path)
    tracks = load_tracks(tracks_path)
    if not motive_rows or any(not tracks[topic] for topic in TOPICS):
        raise ValueError("Motive and both odometry topics are required")

    motive_t = np.asarray(
        [row["capture_realtime_estimate_ns"] for row in motive_rows], dtype=np.int64
    ) * 1e-9
    motive_position = np.asarray(
        [row["position_motive_xyz_m"] for row in motive_rows], dtype=float
    )
    motive_quaternion = np.asarray(
        [row["quaternion_motive_xyzw"] for row in motive_rows], dtype=float
    )
    track_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for topic, rows in tracks.items():
        track_arrays[topic] = (
            np.asarray([row["source_time_ns"] for row in rows], dtype=np.int64) * 1e-9,
            np.asarray([row["position_xyz_m"] for row in rows], dtype=float),
            np.asarray([row["quaternion_xyzw"] for row in rows], dtype=float),
        )

    overlap_start = max(motive_t[0], *(values[0][0] for values in track_arrays.values()))
    overlap_end = min(motive_t[-1], *(values[0][-1] for values in track_arrays.values()))
    if overlap_end - overlap_start < 2.0:
        raise ValueError("less than two seconds of common Motive/odometry overlap")
    query_t = np.arange(overlap_start + 0.2, overlap_end - 0.2, 0.02)
    motive_planar = interp_columns(query_t, motive_t, motive_position[:, [0, 2]])
    motive_yaw = np.interp(query_t, motive_t, yaw_y_up(motive_quaternion))

    scores: dict[str, dict[str, Any]] = {}
    samples: dict[str, dict[str, np.ndarray]] = {}
    for topic, (track_t, track_position, track_quaternion) in track_arrays.items():
        scores[topic], samples[topic] = score_track(
            query_t=query_t,
            motive_planar=motive_planar,
            motive_yaw=motive_yaw,
            track_t=track_t,
            track_position=track_position,
            track_quaternion=track_quaternion,
        )

    sensor = scores["/state_estimation"]
    pelvis = scores["/pelvis_state_estimation"]
    sensor_xy = interp_columns(
        query_t, track_arrays["/state_estimation"][0], track_arrays["/state_estimation"][1][:, :2]
    )
    pelvis_xy = interp_columns(
        query_t,
        track_arrays["/pelvis_state_estimation"][0],
        track_arrays["/pelvis_state_estimation"][1][:, :2],
    )
    sensor_pelvis_offset = np.linalg.norm(sensor_xy - pelvis_xy, axis=1)
    offset_change = np.abs(sensor_pelvis_offset - np.median(sensor_pelvis_offset))

    metrics = {
        "schema": "g1_sensor_pelvis_motive_audit_v1",
        "motive_path": str(motive_path),
        "tracks_path": str(tracks_path),
        "alignment_status": (
            "relative only: one constant planar rotation fitted per topic; "
            "no deployment frame calibration or time-lag fit"
        ),
        "overlap_duration_s": float(overlap_end - overlap_start),
        "query_sample_count": int(len(query_t)),
        "raw_motive_frames": raw_motive_count,
        "usable_motive_frames": len(motive_rows),
        "motive_tracking_coverage": len(motive_rows) / raw_motive_count,
        "topics": scores,
        "pelvis_improvement_percent": {
            "planar_position_rmse": relative_improvement(
                sensor["planar_position_error_m"]["rmse"],
                pelvis["planar_position_error_m"]["rmse"],
            ),
            "relative_position_1s_rmse": relative_improvement(
                sensor["relative_position_error_1s_m"]["rmse"],
                pelvis["relative_position_error_1s_m"]["rmse"],
            ),
            "relative_yaw_rmse": relative_improvement(
                sensor["relative_yaw_error_deg"]["rmse"],
                pelvis["relative_yaw_error_deg"]["rmse"],
            ),
        },
        "sensor_to_pelvis_planar_offset_m": distribution(sensor_pelvis_offset),
        "sensor_to_pelvis_offset_change_from_median_m": distribution(offset_change),
        "bridge_events_in_overlap": event_metrics(
            bridge_path, int(overlap_start * 1e9), int(overlap_end * 1e9)
        ),
    }

    sample_rows: list[dict[str, float]] = []
    motive_relative = motive_planar - motive_planar[0]
    for index, timestamp in enumerate(query_t):
        sample_rows.append(
            {
                "time_from_overlap_s": float(timestamp - overlap_start),
                "motive_x_m": float(motive_relative[index, 0]),
                "motive_y_m": float(motive_relative[index, 1]),
                "sensor_x_m": float(samples["/state_estimation"]["aligned_planar"][index, 0]),
                "sensor_y_m": float(samples["/state_estimation"]["aligned_planar"][index, 1]),
                "pelvis_x_m": float(samples["/pelvis_state_estimation"]["aligned_planar"][index, 0]),
                "pelvis_y_m": float(samples["/pelvis_state_estimation"]["aligned_planar"][index, 1]),
                "sensor_position_error_m": float(samples["/state_estimation"]["position_error"][index]),
                "pelvis_position_error_m": float(samples["/pelvis_state_estimation"]["position_error"][index]),
                "sensor_yaw_error_deg": float(samples["/state_estimation"]["yaw_error_deg"][index]),
                "pelvis_yaw_error_deg": float(samples["/pelvis_state_estimation"]["yaw_error_deg"][index]),
            }
        )
    return metrics, sample_rows


def write_csv(path: Path, rows: Iterable[dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write an empty sample table")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motive", required=True, type=Path)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--bridge-status", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    args = parser.parse_args()
    if args.metrics.exists() or args.samples.exists():
        raise FileExistsError("refusing to overwrite score artifacts")
    metrics, samples = analyze_trial(
        motive_path=args.motive,
        tracks_path=args.tracks,
        bridge_path=args.bridge_status,
    )
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.samples.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.samples, samples)
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
