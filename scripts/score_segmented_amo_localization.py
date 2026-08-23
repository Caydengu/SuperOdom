#!/usr/bin/env python3
"""Score frozen AMO localization windows with one alignment per full run.

This scorer is rate-agnostic so the same windows can compare scan-rate LiDAR
odometry, propagated SuperOdometry state, and root-aware FK treatments. Motive
is evaluator-only: one yaw rotation is fitted on the complete active run and
shared by every method; each frozen segment is only origin-reset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from g1_root_state_bridge.amo_dataset import MotiveCapture, load_lowstate, load_motive
from g1_root_state_bridge.amo_scoring import (
    _distribution,
    _interpolate_reference,
    _robotics_yaw_z_up,
    _rpe,
    _shortest,
    calibrate_score_window_rotation,
    load_treatment_metadata,
    load_treatment_tracks,
    provisional_reference,
)


def _nominal_rate_hz(event_ns: np.ndarray) -> float:
    intervals = np.diff(np.asarray(event_ns, dtype=np.int64)).astype(np.float64) * 1e-9
    intervals = intervals[(intervals > 0.0) & np.isfinite(intervals)]
    if not intervals.size:
        raise ValueError("track has no positive timestamp interval")
    return float(1.0 / np.median(intervals))


def score_segment(
    track: dict[str, np.ndarray],
    motive: MotiveCapture,
    *,
    start_realtime_ns: int,
    end_realtime_ns: int,
    front_plane_to_pelvis_x_m: float,
    estimator_to_reference_rotation: np.ndarray,
    nominal_rate_hz: float | None = None,
) -> dict[str, object]:
    """Score one immutable [start, end) interval without fitting alignment."""

    reference_time, reference_position, reference_yaw = provisional_reference(
        motive, front_plane_to_pelvis_x_m=front_plane_to_pelvis_x_m
    )
    event_all = track["event_realtime_ns"]
    admitted = (
        (event_all >= start_realtime_ns)
        & (event_all < end_realtime_ns)
        & (event_all >= reference_time[0])
        & (event_all <= reference_time[-1])
    )
    event = event_all[admitted]
    if event.size < 3:
        raise ValueError("fewer than three treatment samples overlap frozen segment")
    position = track["position_xyz_m"][admitted]
    quaternion = track["quaternion_xyzw"][admitted]
    ref_position, ref_yaw = _interpolate_reference(
        reference_time, reference_position, reference_yaw, event
    )
    pred_yaw = _robotics_yaw_z_up(quaternion)

    pred_relative = position - position[0]
    ref_relative = ref_position - ref_position[0]
    pred_xy = pred_relative[:, :2] @ estimator_to_reference_rotation
    ref_xz = np.column_stack((ref_relative[:, 0], -ref_relative[:, 2]))
    pred_yaw_relative = pred_yaw - pred_yaw[0]
    ref_yaw_relative = ref_yaw - ref_yaw[0]

    planar_error = np.linalg.norm(pred_xy - ref_xz, axis=1)
    height_error = np.abs(pred_relative[:, 2] - ref_relative[:, 1])
    yaw_error = np.abs(_shortest(pred_yaw_relative - ref_yaw_relative))
    gaps_s = np.diff(event).astype(np.float64) * 1e-9
    rate = nominal_rate_hz or _nominal_rate_hz(event_all)
    expected = max(1, round((end_realtime_ns - start_realtime_ns) * 1e-9 * rate))
    path_length = float(np.sum(np.linalg.norm(np.diff(ref_xz, axis=0), axis=1)))
    terminal = float(planar_error[-1])
    observed_duration_s = float((event[-1] - event[0]) * 1e-9)
    return {
        "sample_count": int(event.size),
        "nominal_rate_hz": rate,
        "availability_fraction": min(1.0, event.size / expected),
        "initialization_delay_ms": max(0.0, (int(event[0]) - start_realtime_ns) * 1e-6),
        "maximum_gap_ms": float(np.max(gaps_s) * 1e3),
        "planar_error_m": _distribution(planar_error),
        "height_error_m": _distribution(height_error),
        "yaw_error_deg": _distribution(np.degrees(yaw_error)),
        "terminal_planar_error_m": terminal,
        "reference_path_length_m": path_length,
        "terminal_drift_percent_of_path": (
            100.0 * terminal / path_length if path_length > 0.01 else None
        ),
        "terminal_drift_m_per_min": (
            terminal / observed_duration_s * 60.0 if observed_duration_s > 0.0 else None
        ),
        "rpe": {
            str(horizon): _rpe(
                event,
                pred_xy,
                ref_xz,
                pred_yaw_relative,
                ref_yaw_relative,
                horizon,
            )
            for horizon in (1.0, 5.0, 10.0)
        },
        "fusion_healthy_fraction": float(np.mean(track["fusion_healthy"][admitted])),
    }


def _aggregate(segments: list[dict[str, Any]]) -> dict[str, object]:
    output: dict[str, object] = {"segment_count": len(segments)}
    for metric, path in {
        "planar_rmse_m": ("metrics", "planar_error_m", "rmse"),
        "yaw_rmse_deg": ("metrics", "yaw_error_deg", "rmse"),
        "rpe_1s_translation_rmse_m": (
            "metrics",
            "rpe",
            "1.0",
            "translation_error_m",
            "rmse",
        ),
        "availability_fraction": ("metrics", "availability_fraction"),
    }.items():
        values = []
        for segment in segments:
            value: Any = segment
            for key in path:
                value = value[key]
            if value is not None:
                values.append(float(value))
        output[metric] = _distribution(np.asarray(values, dtype=np.float64))
    return output


def score_frozen_windows(
    *,
    treatment_path: Path,
    motive_path: Path,
    lowstate_path: Path,
    windows_path: Path,
    run_name: str,
    front_plane_to_pelvis_x_m: float,
    reference_label: str,
) -> dict[str, object]:
    metadata = load_treatment_metadata(treatment_path)
    tracks = load_treatment_tracks(treatment_path)
    capture_manifest_path = motive_path.parent.parent / "manifest.json"
    if not capture_manifest_path.is_file():
        raise ValueError(
            "Motive capture manifest is required to bind rigid-body identity"
        )
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    motive = load_motive(
        motive_path,
        expected_rigid_body_id=int(capture_manifest["rigid_body_id"]),
        expected_rigid_body_name=str(capture_manifest["rigid_body_name"]),
    )
    lowstate = load_lowstate(lowstate_path)
    windows_document = json.loads(windows_path.read_text(encoding="utf-8"))
    matches = [run for run in windows_document["runs"] if run["name"] == run_name]
    if len(matches) != 1:
        raise ValueError(f"expected one frozen-window record for {run_name!r}")
    run = matches[0]
    run_origin_ns = int(lowstate.oslo_event_ns[0])
    scoring_start_ns = run_origin_ns + round(float(run["boundary_s"]) * 1e9)
    sensor = tracks.get("superodom_sensor")
    if sensor is None:
        raise ValueError("one shared alignment requires superodom_sensor")
    rotation, alignment = calibrate_score_window_rotation(
        sensor,
        motive,
        scoring_start_realtime_ns=scoring_start_ns,
        front_plane_to_pelvis_x_m=front_plane_to_pelvis_x_m,
    )
    rates = {name: _nominal_rate_hz(track["event_realtime_ns"]) for name, track in tracks.items()}

    segment_rows: list[dict[str, object]] = []
    for window in run["windows"]:
        start_ns = run_origin_ns + round(float(window["start_s"]) * 1e9)
        end_ns = run_origin_ns + round(float(window["end_s"]) * 1e9)
        for treatment, track in tracks.items():
            segment_rows.append(
                {
                    "run_name": run_name,
                    "window_id": window["window_id"],
                    "content": window["content"],
                    "admission": window["admission"],
                    "start_realtime_ns": start_ns,
                    "end_realtime_ns": end_ns,
                    "treatment": treatment,
                    "metrics": score_segment(
                        track,
                        motive,
                        start_realtime_ns=start_ns,
                        end_realtime_ns=end_ns,
                        front_plane_to_pelvis_x_m=front_plane_to_pelvis_x_m,
                        estimator_to_reference_rotation=rotation,
                        nominal_rate_hz=rates[treatment],
                    ),
                }
            )

    aggregates: dict[str, object] = {}
    for treatment in tracks:
        aggregates[treatment] = {}
        for admission in ("primary", "masked_secondary", "all"):
            selected = [
                row
                for row in segment_rows
                if row["treatment"] == treatment
                and (admission == "all" or row["admission"] == admission)
            ]
            aggregates[treatment][admission] = _aggregate(selected)
        contents = sorted({str(row["content"]) for row in segment_rows})
        aggregates[treatment]["by_content"] = {
            content: _aggregate(
                [
                    row
                    for row in segment_rows
                    if row["treatment"] == treatment and row["content"] == content
                ]
            )
            for content in contents
        }
    return {
        "schema": "g1_amo_segmented_localization_score_v1",
        "run_name": run_name,
        "reference_role": "evaluator_only",
        "reference_label": reference_label,
        "motive_rigid_body_id": int(capture_manifest["rigid_body_id"]),
        "motive_rigid_body_name": str(capture_manifest["rigid_body_name"]),
        "front_plane_to_pelvis_x_m": front_plane_to_pelvis_x_m,
        "absolute_pelvis_front_plane_claims_admitted": False,
        "alignment": alignment,
        "alignment_contract": "one full-active-run yaw rotation shared by every treatment; segment origin reset only",
        "window_contract": windows_document["contract"],
        "treatment_metadata": metadata,
        "nominal_rates_hz": rates,
        "segments": segment_rows,
        "aggregates": aggregates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatments", type=Path, required=True)
    parser.add_argument("--motive", type=Path, required=True)
    parser.add_argument("--lowstate", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--front-plane-to-pelvis-x-m", type=float, default=0.0)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = score_frozen_windows(
        treatment_path=args.treatments,
        motive_path=args.motive,
        lowstate_path=args.lowstate,
        windows_path=args.windows,
        run_name=args.run_name,
        front_plane_to_pelvis_x_m=args.front_plane_to_pelvis_x_m,
        reference_label=args.reference_label,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "segments": len(report["segments"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
