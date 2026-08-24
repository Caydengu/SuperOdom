#!/usr/bin/env python3
"""Separate live stationary heading drift into bias, gap, and sensor effects."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from g1_root_state_bridge.clock_sync import fit_affine_lower_envelope_clock
from g1_root_state_bridge.dynamic_capture_io import iter_recorded_datagrams
from score_stationary_live_localization import (
    load_motive,
    motive_y_up_yaw,
    shortest,
)


def distribution(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


def load_livox(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("kind") == "sample":
                rows.append(row)
    source = np.asarray([row["source_time_ns"] for row in rows], dtype=np.int64)
    receipt = np.asarray([row["receipt_time_ns"] for row in rows], dtype=np.int64)
    gyro = np.asarray([row["angular_velocity_radps"] for row in rows], dtype=np.float64)
    clock = fit_affine_lower_envelope_clock(source, receipt, lower_quantile=0.01)
    event = np.asarray([clock.map_time_ns(int(value)) for value in source], dtype=np.int64)
    return event, receipt, gyro


def load_pelvis(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = list(iter_recorded_datagrams(path))
    source = np.asarray([record.packet.robot_stamp_ns for record in records], dtype=np.int64)
    receipt = np.asarray([record.receipt_realtime_ns for record in records], dtype=np.int64)
    gyro = np.asarray([record.packet.imu_gyroscope for record in records], dtype=np.float64)
    clock = fit_affine_lower_envelope_clock(source, receipt, lower_quantile=0.01)
    event = np.asarray([clock.map_time_ns(int(value)) for value in source], dtype=np.int64)
    return event, receipt, gyro


def integrate_at(
    event_ns: np.ndarray,
    rate_radps: np.ndarray,
    query_ns: np.ndarray,
    *,
    bias_radps: float,
    sign: float,
) -> np.ndarray:
    admitted = (event_ns >= query_ns[0]) & (event_ns <= query_ns[-1])
    time = event_ns[admitted]
    rate = sign * (rate_radps[admitted] - bias_radps)
    dt = np.diff(time).astype(np.float64) * 1e-9
    yaw = np.zeros(time.size, dtype=np.float64)
    yaw[1:] = np.cumsum(0.5 * (rate[:-1] + rate[1:]) * dt)
    origin = int(time[0])
    return np.interp(
        (query_ns - origin).astype(np.float64) * 1e-9,
        (time - origin).astype(np.float64) * 1e-9,
        yaw,
    )


def candidate_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    error_deg = np.degrees(np.abs(shortest(candidate - reference)))
    return {
        "terminal_yaw_deg": float(math.degrees(candidate[-1])),
        "terminal_error_deg": float(error_deg[-1]),
        "rmse_error_deg": float(np.sqrt(np.mean(error_deg**2))),
        "p95_error_deg": float(np.quantile(error_deg, 0.95)),
        "maximum_error_deg": float(np.max(error_deg)),
    }


def startup_bias(event_ns: np.ndarray, gyro_z: np.ndarray, duration_sec: float) -> float:
    admitted = event_ns <= event_ns[0] + round(duration_sec * 1e9)
    if np.count_nonzero(admitted) < 100:
        raise ValueError("startup bias window has too few samples")
    return float(np.mean(gyro_z[admitted]))


def analyze(args: argparse.Namespace) -> dict[str, object]:
    livox_event, livox_receipt, livox_gyro = load_livox(args.livox_imu)
    pelvis_event, pelvis_receipt, pelvis_gyro = load_pelvis(args.lowstate)
    motive = load_motive(args.motive, 42, "G1_PELVIS_F_4123")
    start_ns = max(
        int(livox_event[0]),
        int(pelvis_event[0]),
        int(motive["event_ns"][0]),
    ) + round(args.warmup_sec * 1e9)
    end_ns = min(
        int(livox_event[-1]),
        int(pelvis_event[-1]),
        int(motive["event_ns"][-1]),
    )
    query_ns = np.arange(start_ns, end_ns, 10_000_000, dtype=np.int64)
    motive_seconds = (
        motive["event_ns"] - motive["event_ns"][0]
    ).astype(np.float64) * 1e-9
    query_seconds = (query_ns - motive["event_ns"][0]).astype(np.float64) * 1e-9
    motive_yaw_all = np.unwrap(
        np.asarray([motive_y_up_yaw(value) for value in motive["quaternion"]])
    )
    reference = np.interp(query_seconds, motive_seconds, motive_yaw_all)
    reference -= reference[0]
    livox_startup_bias = startup_bias(
        livox_event,
        livox_gyro[:, 2],
        args.warmup_sec,
    )
    pelvis_startup_bias = startup_bias(
        pelvis_event,
        pelvis_gyro[:, 2],
        args.warmup_sec,
    )
    configured = integrate_at(
        livox_event,
        livox_gyro[:, 2],
        query_ns,
        bias_radps=args.configured_livox_bias_z,
        sign=-1.0,
    )
    livox_startup = integrate_at(
        livox_event,
        livox_gyro[:, 2],
        query_ns,
        bias_radps=livox_startup_bias,
        sign=-1.0,
    )
    pelvis_startup = integrate_at(
        pelvis_event,
        pelvis_gyro[:, 2],
        query_ns,
        bias_radps=pelvis_startup_bias,
        sign=1.0,
    )
    pelvis_startup_reverse = integrate_at(
        pelvis_event,
        pelvis_gyro[:, 2],
        query_ns,
        bias_radps=pelvis_startup_bias,
        sign=-1.0,
    )
    livox_gap_ms = np.diff(livox_event).astype(np.float64) * 1e-6
    pelvis_gap_ms = np.diff(pelvis_event).astype(np.float64) * 1e-6
    return {
        "schema": "g1_stationary_heading_source_analysis_v1",
        "reference_role": "Motive evaluator only",
        "duration_sec": float((query_ns[-1] - query_ns[0]) * 1e-9),
        "configured_livox_bias_z_radps": args.configured_livox_bias_z,
        "startup_bias_window_sec": args.warmup_sec,
        "startup_bias_radps": {
            "livox_z": livox_startup_bias,
            "pelvis_z": pelvis_startup_bias,
            "livox_delta_from_configured": livox_startup_bias
            - args.configured_livox_bias_z,
        },
        "candidates": {
            "livox_fixed_offline_bias": candidate_metrics(configured, reference),
            "livox_startup_stationary_bias": candidate_metrics(livox_startup, reference),
            "pelvis_startup_stationary_bias_positive_z": candidate_metrics(
                pelvis_startup,
                reference,
            ),
            "pelvis_startup_stationary_bias_negative_z": candidate_metrics(
                pelvis_startup_reverse,
                reference,
            ),
        },
        "source_gap_ms": {
            "livox": {
                **distribution(livox_gap_ms),
                "over_25ms": int(np.count_nonzero(livox_gap_ms > 25.0)),
            },
            "pelvis": {
                **distribution(pelvis_gap_ms),
                "over_25ms": int(np.count_nonzero(pelvis_gap_ms > 25.0)),
            },
        },
        "transport_delay_ms": {
            "livox": distribution(
                (livox_receipt - livox_event).astype(np.float64) * 1e-6
            ),
            "pelvis": distribution(
                (pelvis_receipt - pelvis_event).astype(np.float64) * 1e-6
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--livox-imu", type=Path, required=True)
    parser.add_argument("--lowstate", type=Path, required=True)
    parser.add_argument("--motive", type=Path, required=True)
    parser.add_argument("--configured-livox-bias-z", type=float, required=True)
    parser.add_argument("--warmup-sec", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
