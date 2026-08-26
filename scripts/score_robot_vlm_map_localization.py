#!/usr/bin/env python3
"""Score robot-vlm map-frame poses against independent Motive ground truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

try:
    from scripts.validate_motive_polycam_transform import validate_transform
except ModuleNotFoundError:  # Direct execution adds only scripts/ to sys.path.
    from validate_motive_polycam_transform import validate_transform


G1_PELVIS_CONTOUR_MESH_SHA256 = (
    "5cc5c2c7a312329e3feeb2b03d3fc09fc29705bd01864f6767e51be959662420"
)
G1_FRONT_CONTOUR_TO_PELVIS_X_M = -0.061431244015693665


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_motive(
    path: Path,
    *,
    expected_rigid_body_id: int,
    expected_rigid_body_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Load valid NatNet frames and map Motive software time into host realtime."""
    frames = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("record_type") == "metadata":
            if int(row.get("requested_rigid_body_id", -1)) != expected_rigid_body_id:
                raise ValueError("Motive metadata rigid-body ID mismatch")
            if row.get("rigid_body_name") != expected_rigid_body_name:
                raise ValueError("Motive metadata rigid-body name mismatch")
            continue
        if row.get("record_type") != "frame":
            raise ValueError("unsupported Motive row")
        if int(row.get("rigid_body_id", -1)) != expected_rigid_body_id:
            raise ValueError("Motive frame rigid-body ID mismatch")
        if row.get("rigid_body_name") != expected_rigid_body_name:
            raise ValueError("Motive frame rigid-body name mismatch")
        if bool(row.get("tracking_valid")) and float(row["mean_marker_error_m"]) <= 0.005:
            frames.append(row)
    if len(frames) < 3:
        raise ValueError("fewer than three valid Motive frames")
    source = np.rint(
        np.asarray([row["motive_software_time_s"] for row in frames], dtype=np.float64)
        * 1e9
    ).astype(np.int64)
    receipt = np.asarray([row["receipt_realtime_ns"] for row in frames], dtype=np.int64)
    source_origin, receipt_origin = int(source[0]), int(receipt[0])
    x = source.astype(np.float64) - source_origin
    y = receipt.astype(np.float64) - receipt_origin
    x_centered, y_centered = x - np.mean(x), y - np.mean(y)
    scale = float(np.dot(x_centered, y_centered) / np.dot(x_centered, x_centered))
    if not 0.999 <= scale <= 1.001:
        raise ValueError(f"Motive clock scale is implausible: {scale}")
    intercept = float(np.mean(y) - scale * np.mean(x))
    residual = y - (intercept + scale * x)
    lower_envelope = intercept + float(np.quantile(residual, 0.01))
    event = np.rint(receipt_origin + scale * x + lower_envelope).astype(np.int64)
    if not np.all(np.diff(event) > 0):
        raise ValueError("mapped Motive event time is not strictly increasing")
    mapped_residual_ns = receipt - event
    return (
        event,
        np.asarray([row["position_xyz_m_motive_native"] for row in frames], dtype=np.float64),
        np.asarray([row["quaternion_xyzw_motive_native"] for row in frames], dtype=np.float64),
        {
            "clock_scale": scale,
            "clock_scale_error_ppm": abs(scale - 1.0) * 1e6,
            "transport_residual_p50_ms": float(np.quantile(mapped_residual_ns, 0.50) * 1e-6),
            "transport_residual_p95_ms": float(np.quantile(mapped_residual_ns, 0.95) * 1e-6),
            "transport_residual_maximum_ms": float(np.max(mapped_residual_ns) * 1e-6),
        },
    )


def _rotation_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-8:
        raise ValueError("Motive quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def motive_pelvis_reference_in_map(
    position_xyz_m: np.ndarray,
    quaternion_xyzw: np.ndarray,
    *,
    T_map_motive: np.ndarray,
    front_plane_to_pelvis_x_m: float,
    pelvis_yaw_offset_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform the front-plane rigid body to the physical pelvis/root center."""
    position = np.asarray(position_xyz_m, dtype=np.float64)
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    transform = np.asarray(T_map_motive, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3 or quaternion.shape != (len(position), 4):
        raise ValueError("Motive position/quaternion arrays have incompatible shapes")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_map_motive must be finite 4x4")
    map_position = []
    map_yaw = []
    for point, orientation in zip(position, quaternion):
        rotation = _rotation_from_xyzw(orientation)
        front_plane_map = transform @ np.r_[point, 1.0]
        forward_map = transform[:3, :3] @ rotation[:, 0]
        raw_yaw = math.atan2(float(forward_map[1]), float(forward_map[0]))
        robot_yaw = math.atan2(
            math.sin(raw_yaw + pelvis_yaw_offset_rad),
            math.cos(raw_yaw + pelvis_yaw_offset_rad),
        )
        # The Motive asset origin lies on the front pelvis contour.  Its +X
        # direction need not be robot-forward, so apply the CAD-derived planar
        # lever arm only after the independently calibrated heading offset.
        pelvis_map = front_plane_map.copy()
        pelvis_map[:2] += front_plane_to_pelvis_x_m * np.asarray(
            (math.cos(robot_yaw), math.sin(robot_yaw)), dtype=np.float64
        )
        map_position.append(pelvis_map[:3])
        map_yaw.append(robot_yaw)
    return np.asarray(map_position), np.asarray(map_yaw)


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def score_map_trace(
    trace_rows: list[dict],
    *,
    reference_time_ns: np.ndarray,
    reference_position_xyz_m: np.ndarray,
    reference_yaw_rad: np.ndarray,
    T_map_px: np.ndarray,
    maximum_reference_gap_sec: float = 0.30,
) -> tuple[dict, list[dict]]:
    """Compare absolute map poses; no origin reset, ICP fit, or yaw alignment."""
    if not math.isfinite(maximum_reference_gap_sec) or maximum_reference_gap_sec <= 0.0:
        raise ValueError("maximum Motive reference gap must be finite and positive")
    usable = [
        row
        for row in trace_rows
        if row.get("base_pose_xyyaw") is not None
        and row.get("local_estimate_realtime_ns") is not None
    ]
    if not usable:
        raise ValueError("robot-vlm trace has no source-timestamped map poses")
    query = np.asarray([row["local_estimate_realtime_ns"] for row in usable], dtype=np.int64)
    admitted = (query >= reference_time_ns[0]) & (query <= reference_time_ns[-1])
    usable = [row for row, keep in zip(usable, admitted) if keep]
    query = query[admitted]
    right = np.searchsorted(reference_time_ns, query, side="left")
    exact = (right < len(reference_time_ns)) & (
        reference_time_ns[np.minimum(right, len(reference_time_ns) - 1)] == query
    )
    bracketed = (right > 0) & (right < len(reference_time_ns))
    safe_right = np.minimum(right, len(reference_time_ns) - 1)
    safe_left = np.maximum(right - 1, 0)
    bracketing_gap_s = (
        reference_time_ns[safe_right] - reference_time_ns[safe_left]
    ).astype(np.float64) * 1e-9
    reference_fresh = exact | (
        bracketed & (bracketing_gap_s <= maximum_reference_gap_sec)
    )
    reference_gap_rejected_count = int(np.count_nonzero(~reference_fresh))
    usable = [row for row, keep in zip(usable, reference_fresh) if keep]
    query = query[reference_fresh]
    admitted_bracketing_gap_s = bracketing_gap_s[reference_fresh]
    if len(query) < 3:
        raise ValueError("fewer than three map poses have fresh Motive reference")
    origin = int(reference_time_ns[0])
    x = (reference_time_ns - origin).astype(np.float64) * 1e-9
    q = (query - origin).astype(np.float64) * 1e-9
    reference_position = np.column_stack(
        [np.interp(q, x, reference_position_xyz_m[:, axis]) for axis in range(3)]
    )
    reference_yaw_unwrapped = np.unwrap(reference_yaw_rad)
    reference_yaw = np.interp(q, x, reference_yaw_unwrapped)
    estimated = np.asarray([row["base_pose_xyyaw"] for row in usable], dtype=np.float64)
    planar_error = np.linalg.norm(estimated[:, :2] - reference_position[:, :2], axis=1)
    yaw_error = np.abs(
        np.arctan2(
            np.sin(estimated[:, 2] - reference_yaw),
            np.cos(estimated[:, 2] - reference_yaw),
        )
    )
    estimate_delta = estimated[:, :2] - estimated[0, :2]
    reference_delta = reference_position[:, :2] - reference_position[0, :2]
    estimate_yaw_delta = np.arctan2(
        np.sin(estimated[:, 2] - estimated[0, 2]),
        np.cos(estimated[:, 2] - estimated[0, 2]),
    )
    reference_yaw_delta = np.arctan2(
        np.sin(reference_yaw - reference_yaw[0]),
        np.cos(reference_yaw - reference_yaw[0]),
    )
    unique_source_ns = np.unique(query)
    source_gap_s = np.diff(unique_source_ns).astype(np.float64) * 1e-9
    transform = np.asarray(T_map_px, dtype=np.float64)
    inverse = np.linalg.inv(transform)
    enriched = []
    for row, ref, est, pos_error, angle_error in zip(
        usable, reference_position, estimated, planar_error, yaw_error
    ):
        ref_px = inverse @ np.asarray((ref[0], ref[1], 1.0))
        enriched.append(
            {
                **row,
                "motive_pelvis_map_xyz_m": ref.tolist(),
                "motive_pelvis_map_yaw_rad": float(
                    np.interp(
                        (int(row["local_estimate_realtime_ns"]) - origin) * 1e-9,
                        x,
                        reference_yaw_unwrapped,
                    )
                ),
                "motive_map_pixel_uv": [float(ref_px[0]), float(ref_px[1])],
                "absolute_planar_error_m": float(pos_error),
                "absolute_yaw_error_deg": float(np.degrees(angle_error)),
            }
        )
    return {
        "trace_sample_count": len(trace_rows),
        "valid_pose_count": sum(row.get("base_pose_xyyaw") is not None for row in trace_rows),
        "source_timestamped_pose_count": len(usable),
        "motive_overlap_fraction": len(usable) / max(1, len(trace_rows)),
        "motive_reference_gap_rejected_count": reference_gap_rejected_count,
        "maximum_admitted_motive_bracketing_gap_s": (
            float(np.max(admitted_bracketing_gap_s))
            if len(admitted_bracketing_gap_s)
            else 0.0
        ),
        "absolute_planar_error_m": _distribution(planar_error),
        "absolute_yaw_error_deg": _distribution(np.degrees(yaw_error)),
        "estimate_excursion_m": _distribution(np.linalg.norm(estimate_delta, axis=1)),
        "motive_excursion_m": _distribution(np.linalg.norm(reference_delta, axis=1)),
        "estimate_yaw_excursion_deg": _distribution(np.abs(np.degrees(estimate_yaw_delta))),
        "motive_yaw_excursion_deg": _distribution(np.abs(np.degrees(reference_yaw_delta))),
        "terminal_estimate_planar_drift_m": float(np.linalg.norm(estimate_delta[-1])),
        "terminal_estimate_yaw_drift_deg": float(abs(np.degrees(estimate_yaw_delta[-1]))),
        "maximum_unique_source_gap_s": float(np.max(source_gap_s)) if len(source_gap_s) else 0.0,
        "posthoc_alignment_applied": False,
    }, enriched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--motive-map-transform", type=Path, required=True)
    parser.add_argument("--map-artifact", type=Path, required=True)
    parser.add_argument("--map-artifact-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--enriched-trace", type=Path, required=True)
    parser.add_argument(
        "--front-plane-to-pelvis-x-m",
        type=float,
        default=G1_FRONT_CONTOUR_TO_PELVIS_X_M,
    )
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.98)
    parser.add_argument("--maximum-planar-rmse-m", type=float, default=0.10)
    parser.add_argument("--maximum-planar-p95-m", type=float, default=0.20)
    parser.add_argument("--maximum-planar-error-m", type=float, default=0.25)
    parser.add_argument("--maximum-yaw-rmse-deg", type=float, default=5.0)
    parser.add_argument("--maximum-yaw-p95-deg", type=float, default=10.0)
    parser.add_argument("--maximum-yaw-error-deg", type=float, default=15.0)
    parser.add_argument("--maximum-source-gap-sec", type=float, default=0.30)
    parser.add_argument("--maximum-motive-clock-residual-p95-ms", type=float, default=5.0)
    parser.add_argument("--maximum-motive-reference-gap-sec", type=float, default=0.30)
    parser.add_argument("--stationary-maximum-terminal-drift-m", type=float, default=0.03)
    parser.add_argument("--stationary-maximum-terminal-yaw-drift-deg", type=float, default=2.0)
    args = parser.parse_args()
    run_manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    transform = json.loads(args.motive_map_transform.read_text(encoding="utf-8"))
    if _sha256_file(args.map_artifact) != args.map_artifact_sha256.lower():
        parser.error("robot-vlm map artifact SHA-256 mismatch")
    with np.load(args.map_artifact, allow_pickle=False) as archive:
        T_map_px = np.asarray(archive["T_map_px"], dtype=np.float64)
        structural_sha256 = str(np.asarray(archive["scan_hash"]).item())
        glb_sha256 = str(np.asarray(archive["source_glb_sha256"]).item())
        surface_sha256 = str(np.asarray(archive["surface_sha256"]).item())
    try:
        transform_validation = validate_transform(
            transform,
            expected_glb_sha256=glb_sha256,
            expected_surface_sha256=surface_sha256,
            expected_structural_map_sha256=structural_sha256,
            expected_pelvis_rigid_body_id=int(run_manifest["rigid_body_id"]),
            expected_pelvis_rigid_body_name=str(run_manifest["rigid_body_name"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    recorded_id = transform_validation["content_sha256"]
    pelvis_heading = transform["pelvis_heading"]

    motive_time_ns, motive_position, motive_quaternion, motive_clock = _load_motive(
        args.run_dir / "motive/frames.jsonl",
        expected_rigid_body_id=int(run_manifest["rigid_body_id"]),
        expected_rigid_body_name=str(run_manifest["rigid_body_name"]),
    )
    reference_position, reference_yaw = motive_pelvis_reference_in_map(
        motive_position,
        motive_quaternion,
        T_map_motive=np.asarray(transform["T_map_from_motive"], dtype=np.float64),
        front_plane_to_pelvis_x_m=args.front_plane_to_pelvis_x_m,
        pelvis_yaw_offset_rad=float(pelvis_heading["yaw_offset_rad"]),
    )
    trace_rows = [
        json.loads(line)
        for line in (args.run_dir / "traces/robot-vlm-base-pose.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    metrics, enriched = score_map_trace(
        trace_rows,
        reference_time_ns=motive_time_ns,
        reference_position_xyz_m=reference_position,
        reference_yaw_rad=reference_yaw,
        T_map_px=T_map_px,
        maximum_reference_gap_sec=args.maximum_motive_reference_gap_sec,
    )
    failures = []
    if metrics["motive_overlap_fraction"] < args.minimum_overlap_fraction:
        failures.append("motive_overlap")
    if metrics["absolute_planar_error_m"]["rmse"] > args.maximum_planar_rmse_m:
        failures.append("planar_rmse")
    if metrics["absolute_planar_error_m"]["p95"] > args.maximum_planar_p95_m:
        failures.append("planar_p95")
    if metrics["absolute_planar_error_m"]["maximum"] > args.maximum_planar_error_m:
        failures.append("planar_maximum")
    if metrics["absolute_yaw_error_deg"]["rmse"] > args.maximum_yaw_rmse_deg:
        failures.append("yaw_rmse")
    if metrics["absolute_yaw_error_deg"]["p95"] > args.maximum_yaw_p95_deg:
        failures.append("yaw_p95")
    if metrics["absolute_yaw_error_deg"]["maximum"] > args.maximum_yaw_error_deg:
        failures.append("yaw_maximum")
    if metrics["maximum_unique_source_gap_s"] > args.maximum_source_gap_sec:
        failures.append("source_gap")
    if motive_clock["transport_residual_p95_ms"] > args.maximum_motive_clock_residual_p95_ms:
        failures.append("motive_clock_residual")
    if run_manifest.get("capture_class") == "stationary":
        if metrics["terminal_estimate_planar_drift_m"] > args.stationary_maximum_terminal_drift_m:
            failures.append("stationary_terminal_planar_drift")
        if metrics["terminal_estimate_yaw_drift_deg"] > args.stationary_maximum_terminal_yaw_drift_deg:
            failures.append("stationary_terminal_yaw_drift")
    report = {
        "schema": "g1_robot_vlm_absolute_map_localization_score_v1",
        "status": "pass" if not failures else "fail",
        "evidence_class": "independent Motive evaluator of actual robot-vlm map observation",
        "command_capability": "structurally_unavailable",
        "run_dir": str(args.run_dir.resolve()),
        "map_artifact_sha256": args.map_artifact_sha256.lower(),
        "structural_map_sha256": structural_sha256,
        "motive_map_transform_content_sha256": recorded_id,
        "front_plane_to_pelvis_x_m": args.front_plane_to_pelvis_x_m,
        "front_plane_offset_provenance": {
            "robot_identity": "G1-4123",
            "source": "Unitree G1 pelvis_contour_link.STL maximum +X vertex relative to URDF pelvis origin",
            "source_mesh_sha256": G1_PELVIS_CONTOUR_MESH_SHA256,
            "application": "planar offset along independently calibrated robot-forward yaw",
        },
        "motive_clock": motive_clock,
        "metrics": metrics,
        "gates": {
            "minimum_overlap_fraction": args.minimum_overlap_fraction,
            "maximum_planar_rmse_m": args.maximum_planar_rmse_m,
            "maximum_planar_p95_m": args.maximum_planar_p95_m,
            "maximum_planar_error_m": args.maximum_planar_error_m,
            "maximum_yaw_rmse_deg": args.maximum_yaw_rmse_deg,
            "maximum_yaw_p95_deg": args.maximum_yaw_p95_deg,
            "maximum_yaw_error_deg": args.maximum_yaw_error_deg,
            "maximum_source_gap_sec": args.maximum_source_gap_sec,
            "maximum_motive_clock_residual_p95_ms": args.maximum_motive_clock_residual_p95_ms,
            "maximum_motive_reference_gap_sec": args.maximum_motive_reference_gap_sec,
            "stationary_maximum_terminal_drift_m": args.stationary_maximum_terminal_drift_m,
            "stationary_maximum_terminal_yaw_drift_deg": args.stationary_maximum_terminal_yaw_drift_deg,
        },
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.enriched_trace.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.enriched_trace.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in enriched),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
