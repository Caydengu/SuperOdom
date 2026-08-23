#!/usr/bin/env python3
"""Run a pinned KISS-ICP local-odometry baseline on a bounded scan archive."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np


def so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    """Return the SO(3) exponential map of one rotation vector."""
    vector = np.asarray(rotation_vector, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError("rotation vector must have shape (3,)")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        cross = np.asarray(
            ((0.0, -vector[2], vector[1]), (vector[2], 0.0, -vector[0]), (-vector[1], vector[0], 0.0))
        )
        return np.eye(3) + cross
    axis = vector / angle
    cross = np.asarray(
        ((0.0, -axis[2], axis[1]), (axis[2], 0.0, -axis[0]), (-axis[1], axis[0], 0.0))
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def apply_rotation_vectors(
    rotation_vectors: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Apply one Rodrigues rotation vector to each corresponding point."""
    vectors = np.asarray(rotation_vectors, dtype=np.float64)
    values = np.asarray(points, dtype=np.float64)
    if vectors.shape != values.shape or vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("rotation vectors and points must both have shape (N, 3)")
    angles = np.linalg.norm(vectors, axis=1)
    axes = np.zeros_like(vectors)
    nonzero = angles > 1e-12
    axes[nonzero] = vectors[nonzero] / angles[nonzero, None]
    cosine = np.cos(angles)[:, None]
    sine = np.sin(angles)[:, None]
    rotated = values * cosine
    rotated += np.cross(axes, values) * sine
    rotated += axes * np.sum(axes * values, axis=1)[:, None] * (1.0 - cosine)
    rotated[~nonzero] = values[~nonzero]
    return rotated


def _interpolate_vectors(
    time_ns: np.ndarray, vectors: np.ndarray, query_ns: np.ndarray
) -> np.ndarray:
    origin_ns = int(time_ns[0])
    source_s = (np.asarray(time_ns, dtype=np.int64) - origin_ns) * 1e-9
    query_s = (np.asarray(query_ns, dtype=np.int64) - origin_ns) * 1e-9
    return np.column_stack(
        [np.interp(query_s, source_s, vectors[:, axis]) for axis in range(3)]
    )


def gyro_deskew_scan(
    points: np.ndarray,
    relative_time_s: np.ndarray,
    *,
    scan_start_time_ns: int,
    imu_time_ns: np.ndarray,
    imu_angular_velocity_radps: np.ndarray,
    gyro_bias_radps: np.ndarray,
    reference: str = "end",
) -> tuple[np.ndarray, int, float]:
    """Rotate points into the scan start/end frame using the co-mounted IMU.

    Translation during the roughly 100 ms Livox scan is deliberately not
    inferred from the accelerometer.  This treatment isolates whether angular
    distortion from head motion is the dominant local-registration failure.
    """
    xyz = np.asarray(points, dtype=np.float64)
    relative = np.asarray(relative_time_s, dtype=np.float64)
    imu_time = np.asarray(imu_time_ns, dtype=np.int64)
    gyro = np.asarray(imu_angular_velocity_radps, dtype=np.float64)
    bias = np.asarray(gyro_bias_radps, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or relative.shape != (xyz.shape[0],):
        raise ValueError("points must have shape (N,3) and one relative time per point")
    if imu_time.ndim != 1 or gyro.shape != (imu_time.size, 3) or imu_time.size < 2:
        raise ValueError("IMU arrays must have shapes (M,) and (M,3), M >= 2")
    if bias.shape != (3,) or reference not in {"start", "end"}:
        raise ValueError("gyro bias must have shape (3,) and reference must be start/end")
    if np.any(np.diff(imu_time) <= 0):
        raise ValueError("IMU timestamps must be strictly increasing")
    if not np.all(np.isfinite(relative)) or np.min(relative) < -1e-6:
        raise ValueError("point-relative times must be finite and nonnegative")

    point_time_ns = int(scan_start_time_ns) + np.rint(relative * 1e9).astype(np.int64)
    scan_end_time_ns = int(np.max(point_time_ns))
    if int(scan_start_time_ns) < int(imu_time[0]) - 20_000_000:
        raise ValueError("IMU begins more than 20 ms after the scan")
    if scan_end_time_ns > int(imu_time[-1]) + 20_000_000:
        raise ValueError("IMU ends more than 20 ms before the scan")

    interior = imu_time[
        (imu_time > int(scan_start_time_ns)) & (imu_time < scan_end_time_ns)
    ]
    nodes = np.unique(
        np.concatenate(
            (
                np.asarray((int(scan_start_time_ns),), dtype=np.int64),
                interior,
                np.asarray((scan_end_time_ns,), dtype=np.int64),
            )
        )
    )
    midpoint_ns = nodes[:-1] + (nodes[1:] - nodes[:-1]) // 2
    interval_gyro = _interpolate_vectors(imu_time, gyro, midpoint_ns) - bias
    rotations = np.empty((nodes.size, 3, 3), dtype=np.float64)
    rotations[0] = np.eye(3)
    for index in range(nodes.size - 1):
        dt_s = float(nodes[index + 1] - nodes[index]) * 1e-9
        rotations[index + 1] = rotations[index] @ so3_exp(interval_gyro[index] * dt_s)

    segment = np.searchsorted(nodes, point_time_ns, side="right") - 1
    segment = np.clip(segment, 0, nodes.size - 2)
    fractional_dt_s = (point_time_ns - nodes[segment]) * 1e-9
    residual = apply_rotation_vectors(
        interval_gyro[segment] * fractional_dt_s[:, None], xyz
    )
    in_start_frame = np.empty_like(residual)
    for index in np.unique(segment):
        selected = segment == index
        in_start_frame[selected] = residual[selected] @ rotations[index].T
    if reference == "end":
        deskewed = in_start_frame @ rotations[-1]
        reference_time_ns = scan_end_time_ns
    else:
        deskewed = in_start_frame
        reference_time_ns = int(scan_start_time_ns)
    angular_excursion_deg = float(
        np.degrees(np.arccos(np.clip((np.trace(rotations[-1]) - 1.0) / 2.0, -1.0, 1.0)))
    )
    return deskewed, reference_time_ns, angular_excursion_deg


def rotation_to_xyzw(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    quaternion = np.asarray((x, y, z, w), dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def main() -> None:
    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans", type=Path, required=True)
    parser.add_argument("--output-poses", type=Path, required=True)
    parser.add_argument("--output-treatments", type=Path, required=True)
    parser.add_argument("--voxel-size-m", type=float, default=0.15)
    parser.add_argument("--minimum-range-m", type=float, default=0.5)
    parser.add_argument("--maximum-range-m", type=float, default=15.0)
    parser.add_argument("--deskew", action="store_true")
    parser.add_argument(
        "--gyro-deskew",
        action="store_true",
        help="Use archived exact point times and co-mounted IMU angular velocity.",
    )
    parser.add_argument(
        "--gyro-bias-radps",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("GX", "GY", "GZ"),
    )
    parser.add_argument("--deskew-reference", choices=("start", "end"), default="end")
    args = parser.parse_args()
    if args.deskew and args.gyro_deskew:
        parser.error("--deskew and --gyro-deskew are mutually exclusive")
    for output in (args.output_poses, args.output_treatments):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")

    archive = np.load(args.scans, allow_pickle=False)
    points = archive["points_xyz_m"]
    offsets = archive["scan_offsets"]
    source_time_ns = archive["source_time_ns"]
    point_relative_time_s = (
        archive["point_relative_time_s"] if args.gyro_deskew else None
    )
    imu_source_time_ns = archive["imu_source_time_ns"] if args.gyro_deskew else None
    imu_angular_velocity_radps = (
        archive["imu_angular_velocity_radps"] if args.gyro_deskew else None
    )
    config = KISSConfig()
    config.data.min_range = args.minimum_range_m
    config.data.max_range = args.maximum_range_m
    config.data.deskew = args.deskew
    config.mapping.voxel_size = args.voxel_size_m
    odometry = KissICP(config)
    poses = np.empty((source_time_ns.size, 4, 4), dtype=np.float64)
    pose_time_ns = source_time_ns.copy()
    runtimes_ms = np.empty(source_time_ns.size, dtype=np.float64)
    adaptive_threshold = np.empty(source_time_ns.size, dtype=np.float64)
    deskew_angular_excursion_deg = np.zeros(source_time_ns.size, dtype=np.float64)
    deskew_valid = np.ones(source_time_ns.size, dtype=bool)
    deskew_failure_reasons: dict[str, int] = {}
    for index in range(source_time_ns.size):
        scan = np.asarray(points[offsets[index] : offsets[index + 1]], dtype=np.float64)
        if args.gyro_deskew:
            try:
                scan, pose_time_ns[index], deskew_angular_excursion_deg[index] = gyro_deskew_scan(
                    scan,
                    point_relative_time_s[offsets[index] : offsets[index + 1]],
                    scan_start_time_ns=int(source_time_ns[index]),
                    imu_time_ns=imu_source_time_ns,
                    imu_angular_velocity_radps=imu_angular_velocity_radps,
                    gyro_bias_radps=np.asarray(args.gyro_bias_radps, dtype=np.float64),
                    reference=args.deskew_reference,
                )
            except ValueError as error:
                reason = str(error)
                if not reason.startswith(("IMU begins", "IMU ends")):
                    raise
                # Capture boundaries can contain one partial scan without full
                # IMU support. Retain it raw and mark it unavailable instead of
                # extrapolating motion that was never observed.
                deskew_valid[index] = False
                deskew_failure_reasons[reason] = deskew_failure_reasons.get(reason, 0) + 1
        timestamps = (
            np.linspace(0.0, 1.0, scan.shape[0], dtype=np.float64)
            if args.deskew
            else np.zeros(scan.shape[0], dtype=np.float64)
        )
        start = time.perf_counter_ns()
        odometry.register_frame(scan, timestamps)
        runtimes_ms[index] = (time.perf_counter_ns() - start) * 1e-6
        poses[index] = odometry.last_pose
        adaptive_threshold[index] = float(odometry.adaptive_threshold.get_threshold())
        if index % 100 == 0 or index + 1 == source_time_ns.size:
            print(json.dumps({"scan": index + 1, "total": int(source_time_ns.size), "runtime_ms": runtimes_ms[index]}, sort_keys=True), flush=True)

    metadata = {
        "schema": "g1_kiss_icp_replay_v1",
        "source_archive": str(args.scans),
        "kiss_icp_version": importlib.metadata.version("kiss-icp"),
        "motive_online_input": False,
        "voxel_size_m": args.voxel_size_m,
        "minimum_range_m": args.minimum_range_m,
        "maximum_range_m": args.maximum_range_m,
        "deskew": args.deskew,
        "gyro_deskew": args.gyro_deskew,
        "gyro_bias_radps": list(args.gyro_bias_radps),
        "deskew_reference": args.deskew_reference if args.gyro_deskew else None,
        "deskew_timestamp_source": (
            "exact_livox_point_time_and_comounted_imu"
            if args.gyro_deskew
            else "even_scan_order_proxy" if args.deskew else "disabled"
        ),
        "deskew_angular_excursion_deg": {
            "mean": float(np.mean(deskew_angular_excursion_deg)),
            "p95": float(np.quantile(deskew_angular_excursion_deg, 0.95)),
            "maximum": float(np.max(deskew_angular_excursion_deg)),
        },
        "deskew_valid_fraction": float(np.mean(deskew_valid)),
        "deskew_failure_reasons": deskew_failure_reasons,
        "runtime_ms": {
            "mean": float(np.mean(runtimes_ms)),
            "p50": float(np.quantile(runtimes_ms, 0.50)),
            "p95": float(np.quantile(runtimes_ms, 0.95)),
            "p99": float(np.quantile(runtimes_ms, 0.99)),
        },
    }
    args.output_poses.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_poses,
        poses=poses,
        source_time_ns=pose_time_ns,
        runtime_ms=runtimes_ms,
        adaptive_threshold=adaptive_threshold,
        deskew_angular_excursion_deg=deskew_angular_excursion_deg,
        deskew_valid=deskew_valid,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    treatment_name = "kiss_icp_sensor"
    args.output_treatments.parent.mkdir(parents=True, exist_ok=True)
    with args.output_treatments.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({
            "schema": "g1_amo_localization_treatment_track_v1",
            "kind": "metadata",
            "motive_online_input": False,
            "alignment_reference_treatment": treatment_name,
            "kiss_icp": metadata,
        }, sort_keys=True) + "\n")
        for index, pose in enumerate(poses):
            stream.write(json.dumps({
                "schema": "g1_amo_localization_treatment_track_v1",
                "kind": "pose",
                "treatment": treatment_name,
                "source_time_ns": int(pose_time_ns[index]),
                "event_realtime_ns": int(pose_time_ns[index]),
                "source_lowstate_sequence": 0,
                "source_lowstate_age_ns": 0,
                "frame_id": "kiss_icp_local",
                "child_frame_id": "sensor",
                "position_xyz_m": [float(value) for value in pose[:3, 3]],
                "quaternion_xyzw": [float(value) for value in rotation_to_xyzw(pose[:3, :3])],
                "orientation_fusion_healthy": True,
                "orientation_fusion_reason": "kiss_icp_internal_registration",
                "yaw_innovation_rad": 0.0,
                "adaptive_threshold": float(adaptive_threshold[index]),
                "runtime_ms": float(runtimes_ms[index]),
                "deskew_valid": bool(deskew_valid[index]),
            }, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"output_poses": str(args.output_poses), "output_treatments": str(args.output_treatments), **metadata}, sort_keys=True))


if __name__ == "__main__":
    main()
