"""Relative Motive scoring for matched AMO localization treatments.

Absolute pelvis-front-plane claims remain blocked until the fixed Motive
rigid-body transform is independently validated.  This scorer therefore
reports origin-relative motion and labels any mesh-derived lever-arm correction
as provisional sensitivity analysis.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from g1_root_state_bridge.amo_dataset import MotiveCapture, load_motive
from g1_root_state_bridge.amo_treatments import rotation_from_xyzw


def _shortest(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _distribution(values: np.ndarray) -> dict[str, float | int]:
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


def _unwrap(values: np.ndarray) -> np.ndarray:
    return np.unwrap(np.asarray(values, dtype=np.float64))


def _motive_yaw_y_up(quaternion_xyzw: np.ndarray) -> np.ndarray:
    yaw = []
    for quaternion in quaternion_xyzw:
        rotation = rotation_from_xyzw(quaternion)
        forward = rotation[:, 0]
        # Motive is right-handed Y-up.  The corresponding robotics-style
        # horizontal basis is (X, -Z), so positive yaw uses -forward_Z.
        yaw.append(math.atan2(float(-forward[2]), float(forward[0])))
    return _unwrap(np.asarray(yaw))


def _robotics_yaw_z_up(quaternion_xyzw: np.ndarray) -> np.ndarray:
    yaw = []
    for quaternion in quaternion_xyzw:
        rotation = rotation_from_xyzw(quaternion)
        yaw.append(math.atan2(float(rotation[1, 0]), float(rotation[0, 0])))
    return _unwrap(np.asarray(yaw))


def load_treatment_tracks(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            payload = json.loads(line)
            if payload.get("kind") == "metadata":
                continue
            grouped.setdefault(str(payload["treatment"]), []).append(payload)
    tracks: dict[str, dict[str, np.ndarray]] = {}
    for treatment, records in grouped.items():
        event = np.asarray(
            [record["event_realtime_ns"] for record in records], dtype=np.int64
        )
        if not np.all(np.diff(event) > 0):
            raise ValueError(f"{treatment}: event time is not strictly increasing")
        tracks[treatment] = {
            "event_realtime_ns": event,
            "position_xyz_m": np.asarray(
                [record["position_xyz_m"] for record in records], dtype=np.float64
            ),
            "quaternion_xyzw": np.asarray(
                [record["quaternion_xyzw"] for record in records], dtype=np.float64
            ),
            "source_lowstate_age_ns": np.asarray(
                [record["source_lowstate_age_ns"] for record in records], dtype=np.int64
            ),
            "fusion_healthy": np.asarray(
                [
                    bool(record.get("orientation_fusion_healthy", True))
                    for record in records
                ],
                dtype=bool,
            ),
        }
    return tracks


def load_treatment_metadata(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = json.loads(next(stream))
    if payload.get("kind") != "metadata":
        raise ValueError("treatment track is missing its leading metadata record")
    return payload


def provisional_reference(
    motive: MotiveCapture,
    *,
    front_plane_to_pelvis_x_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offset = np.asarray([front_plane_to_pelvis_x_m, 0.0, 0.0], dtype=np.float64)
    positions = []
    for position, quaternion in zip(motive.position_xyz_m, motive.quaternion_xyzw):
        positions.append(position + rotation_from_xyzw(quaternion) @ offset)
    return (
        motive.oslo_event_ns,
        np.asarray(positions, dtype=np.float64),
        _motive_yaw_y_up(motive.quaternion_xyzw),
    )


def _interpolate_reference(
    reference_time_ns: np.ndarray,
    reference_position: np.ndarray,
    reference_yaw: np.ndarray,
    query_time_ns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    origin = int(reference_time_ns[0])
    x = (reference_time_ns - origin).astype(np.float64) * 1e-9
    q = (query_time_ns - origin).astype(np.float64) * 1e-9
    position = np.column_stack(
        [np.interp(q, x, reference_position[:, axis]) for axis in range(3)]
    )
    yaw = np.interp(q, x, reference_yaw)
    return position, yaw


def calibrate_planar_world_rotation(
    sensor_track: dict[str, np.ndarray],
    motive: MotiveCapture,
    *,
    scoring_start_realtime_ns: int,
    front_plane_to_pelvis_x_m: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit one yaw-only world transform on the excluded pre-walk interval."""

    reference_time, reference_position, reference_yaw = provisional_reference(
        motive, front_plane_to_pelvis_x_m=front_plane_to_pelvis_x_m
    )
    event = sensor_track["event_realtime_ns"]
    admitted = (
        (event < scoring_start_realtime_ns)
        & (event >= reference_time[0])
        & (event <= reference_time[-1])
    )
    event = event[admitted]
    sensor_xy = sensor_track["position_xyz_m"][admitted, :2]
    if event.size < 100:
        raise ValueError("pre-walk world-alignment interval has fewer than 100 samples")
    reference, _ = _interpolate_reference(
        reference_time, reference_position, reference_yaw, event
    )
    reference_xy = np.column_stack((reference[:, 0], -reference[:, 2]))
    # Limit both tracks to 20 Hz so a burst of 200 Hz propagated poses does not
    # dominate a particular part of the calibration interval.
    stride = max(1, round(event.size / max(1.0, (event[-1] - event[0]) * 1e-9 * 20.0)))
    sensor_xy = sensor_xy[::stride]
    reference_xy = reference_xy[::stride]
    sensor_centered = sensor_xy - np.mean(sensor_xy, axis=0)
    reference_centered = reference_xy - np.mean(reference_xy, axis=0)
    u, singular_values, vh = np.linalg.svd(sensor_centered.T @ reference_centered)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    residual = np.linalg.norm(sensor_centered @ rotation - reference_centered, axis=1)
    reference_extent = float(np.max(np.linalg.norm(reference_centered, axis=1)))
    if reference_extent < 0.2:
        raise ValueError("pre-walk world-alignment motion has less than 0.2 m extent")
    return rotation, {
        "method": "shared_sensor_track_yaw_only_procrustes_on_excluded_pre_walk_interval",
        "sample_count": int(sensor_xy.shape[0]),
        "start_realtime_ns": int(event[0]),
        "end_realtime_ns": int(event[-1]),
        "duration_s": float((event[-1] - event[0]) * 1e-9),
        "rotation_deg": float(math.degrees(math.atan2(rotation[0, 1], rotation[0, 0]))),
        "rotation_matrix": [[float(value) for value in row] for row in rotation],
        "translation_rmse_m": float(np.sqrt(np.mean(residual**2))),
        "translation_p95_m": float(np.quantile(residual, 0.95)),
        "reference_extent_m": reference_extent,
        "cross_covariance_singular_values": [float(value) for value in singular_values],
        "scale_fitted": False,
        "score_window_used": False,
    }


def calibrate_score_window_rotation(
    sensor_track: dict[str, np.ndarray],
    motive: MotiveCapture,
    *,
    scoring_start_realtime_ns: int,
    front_plane_to_pelvis_x_m: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Post-hoc evaluator alignment for measuring local odometry drift.

    This is deliberately not a deployable global-localization result: Motive
    defines one shared yaw alignment for every treatment after the run.  It is
    useful for separating local odometry quality from initial room heading.
    """

    reference_time, reference_position, reference_yaw = provisional_reference(
        motive, front_plane_to_pelvis_x_m=front_plane_to_pelvis_x_m
    )
    event = sensor_track["event_realtime_ns"]
    admitted = (
        (event >= scoring_start_realtime_ns)
        & (event >= reference_time[0])
        & (event <= reference_time[-1])
    )
    event = event[admitted]
    sensor_xy = sensor_track["position_xyz_m"][admitted, :2]
    if event.size < 100:
        raise ValueError("score-window alignment interval has fewer than 100 samples")
    reference, _ = _interpolate_reference(
        reference_time, reference_position, reference_yaw, event
    )
    reference_xy = np.column_stack((reference[:, 0], -reference[:, 2]))
    stride = max(
        1,
        round(event.size / max(1.0, (event[-1] - event[0]) * 1e-9 * 20.0)),
    )
    sensor_xy = sensor_xy[::stride]
    reference_xy = reference_xy[::stride]
    sensor_centered = sensor_xy - np.mean(sensor_xy, axis=0)
    reference_centered = reference_xy - np.mean(reference_xy, axis=0)
    u, singular_values, vh = np.linalg.svd(sensor_centered.T @ reference_centered)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    residual = np.linalg.norm(sensor_centered @ rotation - reference_centered, axis=1)
    return rotation, {
        "method": "shared_sensor_track_yaw_only_procrustes_on_score_window_evaluator_only",
        "sample_count": int(sensor_xy.shape[0]),
        "start_realtime_ns": int(event[0]),
        "end_realtime_ns": int(event[-1]),
        "duration_s": float((event[-1] - event[0]) * 1e-9),
        "rotation_deg": float(math.degrees(math.atan2(rotation[0, 1], rotation[0, 0]))),
        "rotation_matrix": [[float(value) for value in row] for row in rotation],
        "translation_rmse_m": float(np.sqrt(np.mean(residual**2))),
        "translation_p95_m": float(np.quantile(residual, 0.95)),
        "cross_covariance_singular_values": [float(value) for value in singular_values],
        "scale_fitted": False,
        "score_window_used": True,
        "deployable_global_localization_claim_admitted": False,
    }


def _rpe(
    time_ns: np.ndarray,
    predicted_xy: np.ndarray,
    reference_xy: np.ndarray,
    predicted_yaw: np.ndarray,
    reference_yaw: np.ndarray,
    horizon_s: float,
) -> dict[str, object]:
    target = time_ns + round(horizon_s * 1e9)
    right = np.searchsorted(time_ns, target, side="left")
    valid = right < time_ns.size
    left = np.flatnonzero(valid)
    right = right[valid]
    timing_error_ms = np.abs(time_ns[right] - target[valid]) * 1e-6
    keep = timing_error_ms <= 15.0
    left, right, timing_error_ms = left[keep], right[keep], timing_error_ms[keep]
    pred_delta = predicted_xy[right] - predicted_xy[left]
    ref_delta = reference_xy[right] - reference_xy[left]
    translation_error = np.linalg.norm(pred_delta - ref_delta, axis=1)
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


def score_track(
    track: dict[str, np.ndarray],
    motive: MotiveCapture,
    *,
    scoring_start_realtime_ns: int,
    front_plane_to_pelvis_x_m: float,
    estimator_to_reference_rotation: np.ndarray,
) -> dict[str, object]:
    reference_time, reference_position, reference_yaw = provisional_reference(
        motive, front_plane_to_pelvis_x_m=front_plane_to_pelvis_x_m
    )
    event = track["event_realtime_ns"]
    admitted = (
        (event >= scoring_start_realtime_ns)
        & (event >= reference_time[0])
        & (event <= reference_time[-1])
    )
    event = event[admitted]
    position = track["position_xyz_m"][admitted]
    quaternion = track["quaternion_xyzw"][admitted]
    if event.size < 3:
        raise ValueError(
            "fewer than three treatment samples overlap the score interval"
        )
    ref_position, ref_yaw = _interpolate_reference(
        reference_time, reference_position, reference_yaw, event
    )
    pred_yaw = _robotics_yaw_z_up(quaternion)

    # The one shared yaw rotation is calibrated from the excluded pre-walk
    # sensor track.  Each treatment is only origin-zeroed at score start.
    pred_relative = position - position[0]
    ref_relative = ref_position - ref_position[0]
    pred_xy = pred_relative[:, :2] @ estimator_to_reference_rotation
    ref_xz = np.column_stack((ref_relative[:, 0], -ref_relative[:, 2]))
    pred_yaw_relative = pred_yaw - pred_yaw[0]
    ref_yaw_relative = ref_yaw - ref_yaw[0]

    planar_error = np.linalg.norm(pred_xy - ref_xz, axis=1)
    height_error = np.abs(pred_relative[:, 2] - ref_relative[:, 1])
    yaw_error = np.abs(_shortest(pred_yaw_relative - ref_yaw_relative))
    dt = np.diff(event) * 1e-9
    # The admitted local-pose contract is 200 Hz.  Median interarrival is a
    # biased denominator for a jittered propagated stream and understated this
    # otherwise complete track by about five percent.
    nominal_rate_hz = 200.0
    expected_start_ns = max(scoring_start_realtime_ns, int(reference_time[0]))
    expected_count = round((event[-1] - expected_start_ns) * 1e-9 * nominal_rate_hz) + 1
    path_length = float(np.sum(np.linalg.norm(np.diff(ref_xz, axis=0), axis=1)))
    terminal_planar = float(planar_error[-1])
    duration_s = float((event[-1] - event[0]) * 1e-9)
    return {
        "sample_count": int(event.size),
        "duration_s": duration_s,
        "availability_fraction": min(1.0, event.size / expected_count),
        "initialization_delay_s": max(0.0, (int(event[0]) - expected_start_ns) * 1e-9),
        "nominal_rate_hz": nominal_rate_hz,
        "maximum_gap_ms": float(np.max(dt) * 1e3),
        "lowstate_match_age_ms": _distribution(
            track["source_lowstate_age_ns"][admitted].astype(np.float64) * 1e-6
        ),
        "fusion_healthy_fraction": float(np.mean(track["fusion_healthy"][admitted])),
        "planar_error_m": _distribution(planar_error),
        "height_error_m": _distribution(height_error),
        "yaw_error_deg": _distribution(np.degrees(yaw_error)),
        "terminal_planar_error_m": terminal_planar,
        "reference_path_length_m": path_length,
        "terminal_drift_percent_of_path": (
            100.0 * terminal_planar / path_length if path_length > 0.0 else None
        ),
        "terminal_drift_m_per_min": terminal_planar / duration_s * 60.0,
        "rpe": {
            str(horizon): _rpe(
                event,
                pred_xy,
                ref_xz,
                pred_yaw_relative,
                ref_yaw_relative,
                horizon,
            )
            for horizon in (1.0, 5.0, 10.0, 30.0)
        },
    }


def score_treatments(
    treatment_path: str | Path,
    motive_path: str | Path,
    *,
    scoring_start_realtime_ns: int,
    front_plane_to_pelvis_x_m: float,
    reference_label: str,
    estimator_to_reference_rotation: np.ndarray | None = None,
    fixed_alignment_metadata: dict[str, object] | None = None,
    score_window_evaluator_alignment: bool = False,
    fixed_event_time_shift_ms: float = 0.0,
) -> dict[str, object]:
    metadata = load_treatment_metadata(treatment_path)
    tracks = load_treatment_tracks(treatment_path)
    shift_ns = round(fixed_event_time_shift_ms * 1e6)
    if shift_ns:
        for track in tracks.values():
            track["event_realtime_ns"] = track["event_realtime_ns"] + shift_ns
    motive = load_motive(motive_path)
    if "superodom_sensor" not in tracks:
        raise ValueError(
            "shared world alignment requires the superodom_sensor treatment"
        )
    if estimator_to_reference_rotation is not None and score_window_evaluator_alignment:
        raise ValueError(
            "fixed and score-window alignment modes are mutually exclusive"
        )
    if score_window_evaluator_alignment:
        world_rotation, world_alignment = calibrate_score_window_rotation(
            tracks["superodom_sensor"],
            motive,
            scoring_start_realtime_ns=scoring_start_realtime_ns,
            front_plane_to_pelvis_x_m=front_plane_to_pelvis_x_m,
        )
    elif estimator_to_reference_rotation is None:
        world_rotation, world_alignment = calibrate_planar_world_rotation(
            tracks["superodom_sensor"],
            motive,
            scoring_start_realtime_ns=scoring_start_realtime_ns,
            front_plane_to_pelvis_x_m=front_plane_to_pelvis_x_m,
        )
    else:
        world_rotation = np.asarray(estimator_to_reference_rotation, dtype=np.float64)
        if world_rotation.shape != (2, 2) or not np.allclose(
            world_rotation.T @ world_rotation, np.eye(2), atol=1e-6
        ):
            raise ValueError(
                "fixed estimator-to-reference rotation is not orthonormal 2x2"
            )
        world_alignment = {
            "method": "fixed_yaw_only_rotation_from_development_run",
            "rotation_deg": float(
                math.degrees(math.atan2(world_rotation[0, 1], world_rotation[0, 0]))
            ),
            "rotation_matrix": [
                [float(value) for value in row] for row in world_rotation
            ],
            "scale_fitted": False,
            "score_window_used": False,
            "fixed_alignment_metadata": fixed_alignment_metadata or {},
        }
    return {
        "schema": "g1_motive_amo_relative_localization_score_v1",
        "reference_role": "evaluator_only",
        "reference_label": reference_label,
        "front_plane_to_pelvis_x_m": front_plane_to_pelvis_x_m,
        "absolute_pelvis_front_plane_claims_admitted": False,
        "method_specific_alignment": False,
        "fixed_event_time_shift_ms": fixed_event_time_shift_ms,
        "input_health": {
            "source_odometry": metadata.get("source_odometry_health", {}),
        },
        "alignment": world_alignment,
        "scoring_start_realtime_ns": scoring_start_realtime_ns,
        "treatments": {
            name: score_track(
                track,
                motive,
                scoring_start_realtime_ns=scoring_start_realtime_ns,
                front_plane_to_pelvis_x_m=front_plane_to_pelvis_x_m,
                estimator_to_reference_rotation=world_rotation,
            )
            for name, track in tracks.items()
        },
    }
