#!/usr/bin/env python3
"""Calibrate G1-4123 IMU yaw-rate signs and clocks against Motive.

Motive is used only as an offline evaluator/calibration reference. The output
does not modify raw data and does not estimate a map pose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from g1_root_state_bridge.amo_dataset import (
    detect_walking_boundary,
    fit_affine_lower_envelope_clock,
    load_lowstate,
    load_motive,
)
from g1_root_state_bridge.amo_scoring import _motive_yaw_y_up


def _nearest_distance(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    right = np.clip(np.searchsorted(reference, query), 0, reference.size - 1)
    left = np.clip(right - 1, 0, reference.size - 1)
    return np.minimum(np.abs(query - reference[left]), np.abs(query - reference[right]))


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values.copy()
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")


def _best_axis_lag(
    grid_ns: np.ndarray,
    reference_rate: np.ndarray,
    samples: np.ndarray,
    *,
    maximum_lag_ms: int = 200,
) -> dict[str, object]:
    dt_ns = int(np.median(np.diff(grid_ns)))
    maximum_steps = round(maximum_lag_ms * 1e6 / dt_ns)
    rows: list[dict[str, float | int]] = []
    for axis in range(3):
        for lag_steps in range(-maximum_steps, maximum_steps + 1):
            if lag_steps < 0:
                ref = reference_rate[-lag_steps:]
                candidate = samples[:lag_steps, axis]
            elif lag_steps > 0:
                ref = reference_rate[:-lag_steps]
                candidate = samples[lag_steps:, axis]
            else:
                ref = reference_rate
                candidate = samples[:, axis]
            finite = np.isfinite(ref) & np.isfinite(candidate)
            if int(np.sum(finite)) < 100:
                continue
            ref = ref[finite] - np.mean(ref[finite])
            candidate = candidate[finite] - np.mean(candidate[finite])
            denominator = float(np.linalg.norm(ref) * np.linalg.norm(candidate))
            if denominator <= 0.0:
                continue
            correlation = float(np.dot(ref, candidate) / denominator)
            slope = float(np.dot(candidate, ref) / np.dot(candidate, candidate))
            rows.append(
                {
                    "axis": axis,
                    "lag_ms": lag_steps * dt_ns * 1e-6,
                    "correlation": correlation,
                    "absolute_correlation": abs(correlation),
                    "reference_per_sensor_slope": slope,
                    "sample_count": int(np.sum(finite)),
                }
            )
    if not rows:
        raise ValueError("no finite axis/lag comparison")
    best = max(rows, key=lambda row: float(row["absolute_correlation"]))
    return {
        **best,
        "recommended_sign": 1 if float(best["correlation"]) >= 0.0 else -1,
        "lag_convention": "positive means sensor sample occurs after Motive rate",
    }


def _read_livox_imu(bag: Path, topic_name: str) -> dict[str, np.ndarray]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    header: list[int] = []
    receipt: list[int] = []
    gyro: list[tuple[float, float, float]] = []
    accel: list[tuple[float, float, float]] = []
    orientation: list[tuple[float, float, float, float]] = []
    orientation_covariance: list[tuple[float, ...]] = []
    while reader.has_next():
        topic, serialized, receipt_ns = reader.read_next()
        if topic != topic_name:
            continue
        message = deserialize_message(serialized, Imu)
        header.append(
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        receipt.append(int(receipt_ns))
        gyro.append(
            (
                float(message.angular_velocity.x),
                float(message.angular_velocity.y),
                float(message.angular_velocity.z),
            )
        )
        accel.append(
            (
                float(message.linear_acceleration.x),
                float(message.linear_acceleration.y),
                float(message.linear_acceleration.z),
            )
        )
        orientation.append(
            (
                float(message.orientation.x),
                float(message.orientation.y),
                float(message.orientation.z),
                float(message.orientation.w),
            )
        )
        orientation_covariance.append(tuple(float(v) for v in message.orientation_covariance))
    if len(header) < 3:
        raise ValueError(f"no usable {topic_name} samples in {bag}")
    return {
        "header_ns": np.asarray(header, dtype=np.int64),
        "receipt_ns": np.asarray(receipt, dtype=np.int64),
        "gyro": np.asarray(gyro, dtype=np.float64),
        "accel": np.asarray(accel, dtype=np.float64),
        "orientation": np.asarray(orientation, dtype=np.float64),
        "orientation_covariance": np.asarray(orientation_covariance, dtype=np.float64),
    }


def _distribution(values: np.ndarray) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": [float(value) for value in np.mean(values, axis=0)],
        "std": [float(value) for value in np.std(values, axis=0)],
        "norm_mean": float(np.mean(np.linalg.norm(values, axis=1))),
        "norm_p95": float(np.quantile(np.linalg.norm(values, axis=1), 0.95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imu-topic", default="/utlidar/imu_livox_mid360")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    lowstate = load_lowstate(args.run_dir / "lowstate" / "packets.bin")
    motive = load_motive(
        args.run_dir / "motive" / "frames.jsonl",
        expected_rigid_body_id=int(manifest["rigid_body_id"]),
        expected_rigid_body_name=str(manifest["rigid_body_name"]),
    )
    livox = _read_livox_imu(args.run_dir / "lidar" / "data" / "live_input_probe", args.imu_topic)
    livox_clock = fit_affine_lower_envelope_clock(livox["header_ns"], livox["receipt_ns"])
    livox_event_ns = np.rint(livox_clock.map_ns(livox["header_ns"])).astype(np.int64)
    boundary = detect_walking_boundary(lowstate)

    start_ns = max(
        int(boundary.event_realtime_ns),
        int(motive.oslo_event_ns[0]),
        int(lowstate.oslo_event_ns[0]),
        int(livox_event_ns[0]),
    )
    end_ns = min(
        int(motive.oslo_event_ns[-1]),
        int(lowstate.oslo_event_ns[-1]),
        int(livox_event_ns[-1]),
    )
    step_ns = 10_000_000
    grid_ns = np.arange(start_ns, end_ns, step_ns, dtype=np.int64)
    motive_yaw = _motive_yaw_y_up(motive.quaternion_xyzw)
    motive_yaw_grid = np.interp(grid_ns, motive.oslo_event_ns, motive_yaw)
    motive_rate = _moving_average(np.gradient(motive_yaw_grid, step_ns * 1e-9), 9)
    motive_near = _nearest_distance(motive.oslo_event_ns, grid_ns) <= 20_000_000
    motive_rate[~motive_near] = np.nan

    pelvis_gyro = np.column_stack(
        [
            np.interp(grid_ns, lowstate.oslo_event_ns, lowstate.imu_gyroscope[:, axis])
            for axis in range(3)
        ]
    )
    livox_gyro = np.column_stack(
        [
            np.interp(grid_ns, livox_event_ns, livox["gyro"][:, axis])
            for axis in range(3)
        ]
    )

    stationary_end_ns = int(lowstate.oslo_event_ns[0] + 10_000_000_000)
    pelvis_static = lowstate.oslo_event_ns <= stationary_end_ns
    livox_static = livox_event_ns <= stationary_end_ns
    report = {
        "schema": "g1_4123_imu_motive_alignment_audit_v1",
        "run_dir": str(args.run_dir.resolve()),
        "motive_role": "offline_calibration_and_evaluator_only",
        "raw_data_mutated": False,
        "walking_boundary_s": boundary.seconds_from_lowstate_start,
        "active_overlap_s": (end_ns - start_ns) * 1e-9,
        "motive_grid_coverage": float(np.mean(motive_near)),
        "clock": {
            "pelvis_lowstate": lowstate.clock.as_dict(),
            "livox_imu": livox_clock.as_dict(),
            "motive": motive.clock.as_dict(),
        },
        "stationary_first_10s": {
            "pelvis_gyro_radps": _distribution(lowstate.imu_gyroscope[pelvis_static]),
            "pelvis_accel_mps2": _distribution(lowstate.imu_accelerometer[pelvis_static]),
            "livox_gyro_radps": _distribution(livox["gyro"][livox_static]),
            "livox_accel_mps2": _distribution(livox["accel"][livox_static]),
            "livox_orientation_all_zero_fraction": float(
                np.mean(np.linalg.norm(livox["orientation"][livox_static], axis=1) == 0.0)
            ),
            "livox_orientation_covariance_first": [
                float(v) for v in livox["orientation_covariance"][0]
            ],
        },
        "motive_yaw_rate_match": {
            "pelvis_imu": _best_axis_lag(grid_ns, motive_rate, pelvis_gyro),
            "livox_imu": _best_axis_lag(grid_ns, motive_rate, livox_gyro),
        },
        "admission": {
            "sign_and_axis_fit_admitted": bool(np.mean(motive_near) >= 0.90),
            "absolute_marker_to_pelvis_translation_admitted": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
