"""Causal cross-run structural-map localization for AMO replay datasets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from g1_root_state_bridge.amo_treatments import load_odometry_track, rotation_from_xyzw
from g1_root_state_bridge.waist_kinematics import pelvis_T_mid360


@dataclass(frozen=True)
class ScanArchive:
    points_xyz_m: np.ndarray
    scan_offsets: np.ndarray
    source_time_ns: np.ndarray
    metadata: dict[str, object]

    def scan(self, index: int) -> np.ndarray:
        return self.points_xyz_m[
            self.scan_offsets[index] : self.scan_offsets[index + 1]
        ]


def load_scan_archive(path: str | Path) -> ScanArchive:
    with np.load(path, allow_pickle=False) as archive:
        return ScanArchive(
            points_xyz_m=np.asarray(archive["points_xyz_m"], dtype=np.float32),
            scan_offsets=np.asarray(archive["scan_offsets"], dtype=np.int64),
            source_time_ns=np.asarray(archive["source_time_ns"], dtype=np.int64),
            metadata=json.loads(str(archive["metadata_json"])),
        )


def voxelize_xy(points: np.ndarray, resolution_m: float) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    cells = np.rint(points / resolution_m).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(indices)]


def vertical_persistence_xy(
    points_xyz: np.ndarray,
    *,
    xy_resolution_m: float = 0.10,
    height_resolution_m: float = 0.10,
    minimum_vertical_span_m: float = 0.40,
    minimum_height_bins: int = 4,
) -> np.ndarray:
    """Keep XY columns supported over height, rejecting floors/tabletops."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not points.size:
        raise ValueError("vertical persistence requires a nonempty Nx3 cloud")
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    xy_cell = np.rint(points[:, :2] / xy_resolution_m).astype(np.int64)
    z_cell = np.rint(points[:, 2] / height_resolution_m).astype(np.int64)
    xy_minimum = np.min(xy_cell, axis=0)
    xy_shifted = xy_cell - xy_minimum
    y_stride = int(np.max(xy_shifted[:, 1])) + 1
    xy_key = xy_shifted[:, 0] * y_stride + xy_shifted[:, 1]
    unique_xy_key, inverse = np.unique(xy_key, return_inverse=True)
    group_count = unique_xy_key.size
    z_min = np.full(group_count, np.inf)
    z_max = np.full(group_count, -np.inf)
    np.minimum.at(z_min, inverse, points[:, 2])
    np.maximum.at(z_max, inverse, points[:, 2])
    z_offset = int(np.min(z_cell))
    z_stride = int(np.max(z_cell)) - z_offset + 1
    unique_xy_z = np.unique(inverse.astype(np.int64) * z_stride + z_cell - z_offset)
    height_bin_count = np.bincount(
        unique_xy_z // z_stride,
        minlength=group_count,
    )
    admitted = ((z_max - z_min) >= minimum_vertical_span_m) & (
        height_bin_count >= minimum_height_bins
    )
    if not np.any(admitted):
        raise ValueError("no XY columns pass the vertical-persistence gate")
    xy_sum = np.zeros((group_count, 2), dtype=np.float64)
    xy_count = np.bincount(inverse, minlength=group_count)
    np.add.at(xy_sum, inverse, points[:, :2])
    xy_mean = xy_sum / xy_count[:, None]
    return xy_mean[admitted]


def structural_cloud(
    archive: ScanArchive,
    odometry_path: str | Path | list[dict[str, object]],
    *,
    start_source_time_ns: int,
    end_source_time_ns: int,
    scan_stride: int,
    local_height_min_m: float = -1.1,
    local_height_max_m: float = 1.3,
    voxel_resolution_m: float = 0.10,
    maximum_pose_age_ms: float = 15.0,
) -> tuple[np.ndarray, dict[str, object]]:
    odometry = (
        odometry_path
        if isinstance(odometry_path, list)
        else load_odometry_track(odometry_path).records
    )
    odom_time = np.asarray([row["source_time_ns"] for row in odometry], dtype=np.int64)
    scan_indices = np.flatnonzero(
        (archive.source_time_ns >= start_source_time_ns)
        & (archive.source_time_ns <= end_source_time_ns)
    )[::scan_stride]
    if not scan_indices.size:
        raise ValueError("no scans overlap the structural-cloud window")
    right = np.searchsorted(
        odom_time, archive.source_time_ns[scan_indices], side="left"
    )
    right = np.clip(right, 0, odom_time.size - 1)
    left = np.clip(right - 1, 0, odom_time.size - 1)
    choose_left = np.abs(
        archive.source_time_ns[scan_indices] - odom_time[left]
    ) <= np.abs(odom_time[right] - archive.source_time_ns[scan_indices])
    matches = np.where(choose_left, left, right)
    ages_ns = np.abs(archive.source_time_ns[scan_indices] - odom_time[matches])
    admitted = ages_ns <= maximum_pose_age_ms * 1e6
    transformed: list[np.ndarray] = []
    for scan_index, odom_index in zip(scan_indices[admitted], matches[admitted]):
        points = archive.scan(int(scan_index)).astype(np.float64)
        points = points[
            (points[:, 2] >= local_height_min_m) & (points[:, 2] <= local_height_max_m)
        ]
        row = odometry[int(odom_index)]
        rotation = rotation_from_xyzw(np.asarray(row["quaternion_xyzw"]))
        translation = np.asarray(row["position_xyz_m"], dtype=np.float64)
        world = points @ rotation.T + translation
        transformed.append(world[:, :2])
    if not transformed:
        raise ValueError("no scans passed the pose-age and height gates")
    cloud = voxelize_xy(np.concatenate(transformed, axis=0), voxel_resolution_m)
    return cloud, {
        "requested_scan_count": int(scan_indices.size),
        "admitted_scan_count": int(np.sum(admitted)),
        "pose_match_age_ms_p95": float(np.quantile(ages_ns[admitted], 0.95) * 1e-6),
        "point_count": int(cloud.shape[0]),
        "scan_stride": scan_stride,
        "voxel_resolution_m": voxel_resolution_m,
        "local_height_range_m": [local_height_min_m, local_height_max_m],
        "start_source_time_ns": start_source_time_ns,
        "end_source_time_ns": end_source_time_ns,
    }


def pelvis_structural_cloud(
    archive: ScanArchive,
    pelvis_records: list[dict[str, object]],
    lowstate: object,
    *,
    start_source_time_ns: int,
    end_source_time_ns: int,
    scan_stride: int = 2,
    pelvis_height_min_m: float = -1.1,
    pelvis_height_max_m: float = 1.3,
    voxel_resolution_m: float = 0.10,
    maximum_pose_age_ms: float = 15.0,
    maximum_lowstate_age_ms: float = 10.0,
    minimum_vertical_span_m: float = 0.40,
    minimum_height_bins: int = 4,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build a z-up local structural cloud from pelvis odometry and sensor scans."""
    pose_time = np.asarray(
        [row["source_time_ns"] for row in pelvis_records], dtype=np.int64
    )
    scan_indices = np.flatnonzero(
        (archive.source_time_ns >= start_source_time_ns)
        & (archive.source_time_ns <= end_source_time_ns)
    )[::scan_stride]
    if not scan_indices.size:
        raise ValueError("no scans overlap the pelvis structural-cloud window")

    def nearest(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
        right = np.clip(
            np.searchsorted(reference, query, side="left"), 0, reference.size - 1
        )
        left = np.clip(right - 1, 0, reference.size - 1)
        return np.where(
            np.abs(query - reference[left]) <= np.abs(reference[right] - query),
            left,
            right,
        )

    scan_time = archive.source_time_ns[scan_indices]
    pose_index = nearest(pose_time, scan_time)
    lowstate_time = np.asarray(lowstate.oslo_event_ns, dtype=np.int64)
    lowstate_index = nearest(lowstate_time, scan_time)
    pose_age_ns = np.abs(scan_time - pose_time[pose_index])
    lowstate_age_ns = np.abs(scan_time - lowstate_time[lowstate_index])
    admitted = (pose_age_ns <= maximum_pose_age_ms * 1e6) & (
        lowstate_age_ns <= maximum_lowstate_age_ms * 1e6
    )
    transformed: list[np.ndarray] = []
    for scan_index, pose_match, low_match in zip(
        scan_indices[admitted], pose_index[admitted], lowstate_index[admitted]
    ):
        row = pelvis_records[int(pose_match)]
        world_R_pelvis = rotation_from_xyzw(np.asarray(row["quaternion_xyzw"]))
        world_t_pelvis = np.asarray(row["position_xyz_m"], dtype=np.float64)
        q = np.asarray(lowstate.joint_position[int(low_match)], dtype=np.float64)
        pelvis_T_sensor = pelvis_T_mid360(*q[12:15])
        points = archive.scan(int(scan_index)).astype(np.float64)
        points_pelvis = points @ pelvis_T_sensor[:3, :3].T + pelvis_T_sensor[:3, 3]
        height_gate = (points_pelvis[:, 2] >= pelvis_height_min_m) & (
            points_pelvis[:, 2] <= pelvis_height_max_m
        )
        points_pelvis = points_pelvis[height_gate]
        world = points_pelvis @ world_R_pelvis.T + world_t_pelvis
        transformed.append(world)
    if not transformed:
        raise ValueError("no scans passed pelvis pose, LowState, and height gates")
    cloud = vertical_persistence_xy(
        np.concatenate(transformed, axis=0),
        xy_resolution_m=voxel_resolution_m,
        minimum_vertical_span_m=minimum_vertical_span_m,
        minimum_height_bins=minimum_height_bins,
    )
    return cloud, {
        "requested_scan_count": int(scan_indices.size),
        "admitted_scan_count": int(np.sum(admitted)),
        "pose_match_age_ms_p95": float(np.quantile(pose_age_ns[admitted], 0.95) * 1e-6),
        "lowstate_match_age_ms_p95": float(
            np.quantile(lowstate_age_ns[admitted], 0.95) * 1e-6
        ),
        "point_count": int(cloud.shape[0]),
        "scan_stride": scan_stride,
        "voxel_resolution_m": voxel_resolution_m,
        "pelvis_height_range_m": [pelvis_height_min_m, pelvis_height_max_m],
        "minimum_vertical_span_m": minimum_vertical_span_m,
        "minimum_height_bins": minimum_height_bins,
        "start_source_time_ns": start_source_time_ns,
        "end_source_time_ns": end_source_time_ns,
        "frame_contract": "physical_mid360_points -> dynamic pelvis FK -> initial pelvis local",
    }


def _rotation_row(angle_rad: float) -> np.ndarray:
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    return np.asarray(((cosine, sine), (-sine, cosine)), dtype=np.float64)


def _occupancy(
    points: np.ndarray, resolution_m: float
) -> tuple[np.ndarray, np.ndarray]:
    cells = np.rint(points / resolution_m).astype(np.int64)
    minimum = np.min(cells, axis=0)
    cells = cells - minimum
    grid = np.zeros(tuple(np.max(cells, axis=0) + 1), dtype=np.float32)
    # Repeated advanced-index assignments are idempotent.  Sorting every point
    # through ``np.unique`` produced the same binary occupancy but dominated
    # global-localization latency (187 sorts per initialization on the frozen
    # yaw grid).
    grid[cells[:, 0], cells[:, 1]] = 1.0
    return grid, minimum.astype(np.float64) * resolution_m


def correlate_at_yaw(
    map_xy: np.ndarray,
    query_xy: np.ndarray,
    *,
    angle_rad: float,
    resolution_m: float,
) -> dict[str, object]:
    rotation = _rotation_row(angle_rad)
    rotated = query_xy @ rotation
    map_grid, map_minimum = _occupancy(map_xy, resolution_m)
    query_grid, query_minimum = _occupancy(rotated, resolution_m)
    correlation = fftconvolve(map_grid, query_grid[::-1, ::-1], mode="full")
    flat_index = int(np.argmax(correlation))
    peak_index = np.asarray(
        np.unravel_index(flat_index, correlation.shape), dtype=np.int64
    )
    shift_cells = peak_index - (np.asarray(query_grid.shape, dtype=np.int64) - 1)
    translation = (
        shift_cells.astype(np.float64) * resolution_m + map_minimum - query_minimum
    )
    peak = float(correlation[tuple(peak_index)])
    return {
        "angle_rad": angle_rad,
        "rotation": rotation,
        "translation_m": translation,
        "overlap_cells": peak,
        "overlap_fraction": peak / max(1.0, float(np.sum(query_grid))),
    }


def _fit_rigid_row(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    u, _, vh = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    translation = target_mean - source_mean @ rotation
    return rotation, translation


def directional_observability(
    map_xy: np.ndarray,
    transformed_query_xy: np.ndarray,
    target_indices: np.ndarray,
    admitted: np.ndarray,
    *,
    neighbors: int = 8,
) -> dict[str, object]:
    """Measure planar point-to-plane information in x, y, and yaw.

    Scalar nearest-neighbor residuals can look excellent along a single wall
    while translation tangent to that wall is unobservable.  This normalized
    Gauss-Newton information matrix exposes that directional failure.
    """
    selected = np.flatnonzero(admitted)
    if selected.size < max(100, neighbors):
        return {
            "eigenvalues": [0.0, 0.0, 0.0],
            "minimum_eigenvalue": 0.0,
            "condition_number": math.inf,
            "sample_count": int(selected.size),
            "observable": False,
        }
    tree = cKDTree(map_xy)
    _, neighbor_index = tree.query(
        map_xy[target_indices[selected]], k=min(neighbors, map_xy.shape[0]), workers=-1
    )
    neighborhood = map_xy[np.asarray(neighbor_index)]
    centered = neighborhood - np.mean(neighborhood, axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    del eigenvalues
    normal = eigenvectors[:, :, 0]
    point = transformed_query_xy[selected]
    yaw_column = normal[:, 0] * (-point[:, 1]) + normal[:, 1] * point[:, 0]
    jacobian = np.column_stack((normal[:, 0], normal[:, 1], yaw_column))
    information = jacobian.T @ jacobian / selected.size
    spectrum = np.linalg.eigvalsh(information)
    minimum = float(spectrum[0])
    condition = float(spectrum[-1] / minimum) if minimum > 1e-12 else math.inf
    return {
        "eigenvalues": [float(value) for value in spectrum],
        "minimum_eigenvalue": minimum,
        "condition_number": condition,
        "sample_count": int(selected.size),
        "observable": bool(minimum >= 0.02 and condition <= 1e6),
    }


def refine_icp(
    map_xy: np.ndarray,
    query_xy: np.ndarray,
    rotation: np.ndarray,
    translation_m: np.ndarray,
    *,
    maximum_correspondence_m: float = 0.35,
    iterations: int = 30,
) -> dict[str, object]:
    tree = cKDTree(map_xy)
    total_rotation = np.asarray(rotation, dtype=np.float64)
    total_translation = np.asarray(translation_m, dtype=np.float64)
    previous_rmse = math.inf
    for iteration in range(iterations):
        transformed = query_xy @ total_rotation + total_translation
        distance, index = tree.query(transformed, workers=-1)
        admitted = distance <= maximum_correspondence_m
        if np.sum(admitted) < 100:
            break
        delta_rotation, delta_translation = _fit_rigid_row(
            transformed[admitted], map_xy[index[admitted]]
        )
        total_rotation = total_rotation @ delta_rotation
        total_translation = total_translation @ delta_rotation + delta_translation
        rmse = float(np.sqrt(np.mean(distance[admitted] ** 2)))
        if abs(previous_rmse - rmse) < 1e-5:
            break
        previous_rmse = rmse
    transformed = query_xy @ total_rotation + total_translation
    distance, index = tree.query(transformed, workers=-1)
    admitted = distance <= maximum_correspondence_m
    residual = distance[admitted]
    observability = directional_observability(map_xy, transformed, index, admitted)
    return {
        "rotation": total_rotation,
        "translation_m": total_translation,
        "iterations": iteration + 1,
        "inlier_fraction": float(np.mean(admitted)),
        "inlier_count": int(np.sum(admitted)),
        "rmse_m": float(np.sqrt(np.mean(residual**2))) if residual.size else math.inf,
        "p95_m": float(np.quantile(residual, 0.95)) if residual.size else math.inf,
        "maximum_correspondence_m": maximum_correspondence_m,
        "observability": observability,
    }


def global_localize(
    map_xy: np.ndarray,
    query_xy: np.ndarray,
    *,
    coarse_yaw_step_deg: float = 5.0,
    fine_yaw_step_deg: float = 0.5,
    correlation_resolution_m: float = 0.15,
    maximum_correspondence_m: float = 0.35,
) -> dict[str, object]:
    coarse = [
        correlate_at_yaw(
            map_xy,
            query_xy,
            angle_rad=math.radians(float(angle)),
            resolution_m=correlation_resolution_m,
        )
        for angle in np.arange(-180.0, 180.0, coarse_yaw_step_deg)
    ]
    coarse.sort(key=lambda item: float(item["overlap_fraction"]), reverse=True)
    best_angle_deg = math.degrees(float(coarse[0]["angle_rad"]))
    fine_angles = np.arange(
        best_angle_deg - coarse_yaw_step_deg,
        best_angle_deg + coarse_yaw_step_deg + 0.5 * fine_yaw_step_deg,
        fine_yaw_step_deg,
    )
    fine = [
        correlate_at_yaw(
            map_xy,
            query_xy,
            angle_rad=math.radians(float(angle)),
            resolution_m=correlation_resolution_m,
        )
        for angle in fine_angles
    ]
    fine.sort(key=lambda item: float(item["overlap_fraction"]), reverse=True)
    best = fine[0]
    icp = refine_icp(
        map_xy,
        query_xy,
        np.asarray(best["rotation"]),
        np.asarray(best["translation_m"]),
        maximum_correspondence_m=maximum_correspondence_m,
    )
    second_nonlocal = next(
        (
            item
            for item in coarse[1:]
            if abs(
                (
                    (math.degrees(float(item["angle_rad"])) - best_angle_deg + 180.0)
                    % 360.0
                )
                - 180.0
            )
            >= 15.0
        ),
        coarse[1],
    )
    peak_ratio = float(best["overlap_fraction"]) / max(
        1e-12, float(second_nonlocal["overlap_fraction"])
    )
    healthy = (
        float(best["overlap_fraction"]) >= 0.12
        and peak_ratio >= 1.03
        and float(icp["inlier_fraction"]) >= 0.45
        and float(icp["p95_m"]) <= 0.25
        and bool(icp["observability"]["observable"])
    )
    return {
        "schema": "g1_cross_run_map_localization_v1",
        "rotation_matrix": np.asarray(icp["rotation"]).tolist(),
        "yaw_deg": float(
            math.degrees(math.atan2(icp["rotation"][0, 1], icp["rotation"][0, 0]))
        ),
        "translation_m": np.asarray(icp["translation_m"]).tolist(),
        "correlation": {
            "resolution_m": correlation_resolution_m,
            "best_overlap_fraction": float(best["overlap_fraction"]),
            "best_overlap_cells": float(best["overlap_cells"]),
            "nonlocal_second_overlap_fraction": float(
                second_nonlocal["overlap_fraction"]
            ),
            "peak_ratio": peak_ratio,
            "top_coarse": [
                {
                    "yaw_deg": math.degrees(float(item["angle_rad"])),
                    "overlap_fraction": float(item["overlap_fraction"]),
                    "translation_m": np.asarray(item["translation_m"]).tolist(),
                }
                for item in coarse[:8]
            ],
        },
        "icp": {
            key: value
            for key, value in icp.items()
            if key not in {"rotation", "translation_m"}
        },
        "gates": {
            "minimum_overlap_fraction": 0.12,
            "minimum_peak_ratio": 1.03,
            "minimum_icp_inlier_fraction": 0.45,
            "maximum_icp_p95_m": 0.25,
            "icp_maximum_correspondence_m": maximum_correspondence_m,
            "minimum_observability_eigenvalue": 0.02,
            "maximum_observability_condition_number": 1e6,
        },
        "healthy": healthy,
    }
