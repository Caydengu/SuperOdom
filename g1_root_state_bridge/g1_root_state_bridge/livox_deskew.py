"""Exact point-time rotational deskew for the G1 Mid-360 stream."""

from __future__ import annotations

import numpy as np


class LivoxDeskewError(ValueError):
    """A scan or IMU window cannot support physical point-time deskew."""


def so3_exp(rotation_vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotation_vector, dtype=np.float64)
    if vector.shape != (3,):
        raise LivoxDeskewError("rotation vector must have shape (3,)")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        x, y, z = vector
        return np.eye(3) + np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    axis = vector / angle
    x, y, z = axis
    cross = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _apply_rotation_vectors(rotation_vectors: np.ndarray, points: np.ndarray) -> np.ndarray:
    vectors = np.asarray(rotation_vectors, dtype=np.float64)
    values = np.asarray(points, dtype=np.float64)
    if vectors.shape != values.shape or vectors.ndim != 2 or vectors.shape[1] != 3:
        raise LivoxDeskewError("rotation vectors and points must have shape (N,3)")
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


def _interpolate_vectors(time_ns: np.ndarray, vectors: np.ndarray, query_ns: np.ndarray) -> np.ndarray:
    origin_ns = int(time_ns[0])
    source_s = (np.asarray(time_ns, dtype=np.int64) - origin_ns) * 1e-9
    query_s = (np.asarray(query_ns, dtype=np.int64) - origin_ns) * 1e-9
    return np.column_stack(
        [np.interp(query_s, source_s, vectors[:, axis]) for axis in range(3)]
    )


def gyro_deskew_scan(
    points_xyz_m: np.ndarray,
    point_relative_time_s: np.ndarray,
    *,
    scan_start_time_ns: int,
    imu_time_ns: np.ndarray,
    imu_angular_velocity_radps: np.ndarray,
    gyro_bias_radps: np.ndarray,
    reference: str = "end",
    maximum_boundary_gap_ns: int = 20_000_000,
) -> tuple[np.ndarray, int, float]:
    """Rotate every point into the scan start or end frame.

    Translation during a scan is deliberately not synthesized from noisy
    acceleration.  Missing IMU coverage fails closed rather than extrapolating.
    """

    xyz = np.asarray(points_xyz_m, dtype=np.float64)
    relative = np.asarray(point_relative_time_s, dtype=np.float64)
    imu_time = np.asarray(imu_time_ns, dtype=np.int64)
    gyro = np.asarray(imu_angular_velocity_radps, dtype=np.float64)
    bias = np.asarray(gyro_bias_radps, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or relative.shape != (xyz.shape[0],):
        raise LivoxDeskewError("points and point-relative times have incompatible shapes")
    if xyz.shape[0] == 0 or not np.all(np.isfinite(xyz)):
        raise LivoxDeskewError("scan must contain finite points")
    if imu_time.ndim != 1 or gyro.shape != (imu_time.size, 3) or imu_time.size < 2:
        raise LivoxDeskewError("IMU arrays must contain at least two 3-axis samples")
    if bias.shape != (3,) or not np.all(np.isfinite(bias)):
        raise LivoxDeskewError("gyro bias must contain three finite values")
    if reference not in {"start", "end"}:
        raise LivoxDeskewError("reference must be start or end")
    if np.any(np.diff(imu_time) <= 0):
        raise LivoxDeskewError("IMU timestamps must be strictly increasing")
    if not np.all(np.isfinite(relative)) or np.min(relative) < -1e-6:
        raise LivoxDeskewError("point-relative times must be finite and nonnegative")

    point_time_ns = int(scan_start_time_ns) + np.rint(relative * 1e9).astype(np.int64)
    scan_end_time_ns = int(np.max(point_time_ns))
    if int(scan_start_time_ns) < int(imu_time[0]) - maximum_boundary_gap_ns:
        raise LivoxDeskewError("IMU begins too late for the scan")
    if scan_end_time_ns > int(imu_time[-1]) + maximum_boundary_gap_ns:
        raise LivoxDeskewError("IMU ends too early for the scan")

    interior = imu_time[(imu_time > int(scan_start_time_ns)) & (imu_time < scan_end_time_ns)]
    nodes = np.unique(
        np.concatenate(
            ((int(scan_start_time_ns),), interior, (scan_end_time_ns,))
        ).astype(np.int64)
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
    residual = _apply_rotation_vectors(interval_gyro[segment] * fractional_dt_s[:, None], xyz)
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
        np.degrees(
            np.arccos(np.clip((np.trace(rotations[-1]) - 1.0) / 2.0, -1.0, 1.0))
        )
    )
    return deskewed, reference_time_ns, angular_excursion_deg
