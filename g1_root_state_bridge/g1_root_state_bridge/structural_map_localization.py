"""Automatic, fail-closed Polycam structural-map correction lane.

The fast KISS-local trajectory is never mutated here.  This module consumes
registered ``kiss_local`` point clouds, estimates ``map_T_kiss_local``, and
emits only accepted slow-lane candidates.  Motive is not an input.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from g1_root_state_bridge.map_protocol import (
    MapCorrectionHealth,
    MapCorrectionPacketV1,
    map_transform_from_row_se2,
)


class StructuralMapLocalizationError(ValueError):
    """Map evidence violates a frame, timing, or geometry contract."""


@dataclass(frozen=True)
class StructuralMapConfig:
    global_window_sec: float = 10.0
    tracking_window_sec: float = 5.0
    tracking_period_sec: float = 2.0
    correlation_resolution_m: float = 0.15
    global_maximum_correspondence_m: float = 0.25
    tracking_maximum_correspondence_m: float = 0.25
    minimum_overlap_fraction: float = 0.12
    minimum_peak_ratio: float = 1.03
    minimum_inlier_fraction: float = 0.45
    maximum_global_p95_m: float = 0.25
    maximum_rmse_m: float = 0.12
    maximum_tracking_p95_m: float = 0.22
    minimum_observability_eigenvalue: float = 0.02
    maximum_observability_condition_number: float = 1e6
    maximum_pose_jump_m: float = 0.25
    maximum_yaw_jump_deg: float = 5.0
    maximum_input_age_ns: int = 500_000_000
    maximum_root_evidence_gap_ns: int = 150_000_000
    local_height_min_m: float = -1.1
    local_height_max_m: float = 1.3
    query_voxel_resolution_m: float = 0.10
    minimum_vertical_span_m: float = 0.40
    minimum_height_bins: int = 4


@dataclass(frozen=True)
class TimedRegisteredCloud:
    source_time_ns: int
    frame_id: str
    points_xyz_m: np.ndarray


@dataclass(frozen=True)
class TimedLocalPose:
    source_time_ns: int
    frame_id: str
    position_xy_m: np.ndarray


@dataclass(frozen=True)
class MapCorrectionAttempt:
    kind: str
    accepted: bool
    rejection_reason: str | None
    reference_time_ns: int
    evidence_time_ns: int
    runtime_ms: float
    report: dict[str, object]
    packet: MapCorrectionPacketV1 | None


class RegisteredMapEvidenceBuffer:
    """Bounded source-time buffer for matched KISS-local ROS streams."""

    def __init__(self, *, maximum_history_sec: float = 15.0) -> None:
        if maximum_history_sec <= 0.0:
            raise StructuralMapLocalizationError("history duration must be positive")
        self.maximum_history_ns = round(maximum_history_sec * 1e9)
        self.clouds: list[TimedRegisteredCloud] = []
        self.poses: list[TimedLocalPose] = []

    def clear(self) -> None:
        self.clouds.clear()
        self.poses.clear()

    def _trim(self, newest_ns: int) -> None:
        cutoff = newest_ns - self.maximum_history_ns
        self.clouds = [row for row in self.clouds if row.source_time_ns >= cutoff]
        self.poses = [row for row in self.poses if row.source_time_ns >= cutoff]

    def append_cloud(self, row: TimedRegisteredCloud) -> None:
        points = np.asarray(row.points_xyz_m, dtype=np.float64)
        if row.source_time_ns <= 0 or row.frame_id != "kiss_local":
            raise StructuralMapLocalizationError(
                "registered cloud must be source-stamped in kiss_local"
            )
        if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise StructuralMapLocalizationError(
                "registered cloud must contain finite Nx3 points"
            )
        if self.clouds and row.source_time_ns <= self.clouds[-1].source_time_ns:
            raise StructuralMapLocalizationError(
                "registered cloud source time did not increase"
            )
        self.clouds.append(
            TimedRegisteredCloud(row.source_time_ns, row.frame_id, points.copy())
        )
        self._trim(row.source_time_ns)

    def append_pose(self, row: TimedLocalPose) -> None:
        position = np.asarray(row.position_xy_m, dtype=np.float64)
        if row.source_time_ns <= 0 or row.frame_id != "kiss_local":
            raise StructuralMapLocalizationError(
                "local pose must be source-stamped in kiss_local"
            )
        if position.shape != (2,) or not np.all(np.isfinite(position)):
            raise StructuralMapLocalizationError(
                "local pose position must be a finite 2-vector"
            )
        if self.poses and row.source_time_ns <= self.poses[-1].source_time_ns:
            raise StructuralMapLocalizationError(
                "local pose source time did not increase"
            )
        self.poses.append(
            TimedLocalPose(row.source_time_ns, row.frame_id, position.copy())
        )
        self._trim(row.source_time_ns)

    def structural_query(
        self,
        *,
        end_source_time_ns: int,
        window_sec: float,
        config: StructuralMapConfig,
    ) -> tuple[np.ndarray, dict[str, object]]:
        start_ns = end_source_time_ns - round(window_sec * 1e9)
        selected = [
            row
            for row in self.clouds
            if start_ns <= row.source_time_ns <= end_source_time_ns
        ]
        if len(selected) < 3:
            raise StructuralMapLocalizationError(
                "fewer than three registered scans overlap the query window"
            )
        if selected[0].source_time_ns > start_ns + 250_000_000:
            raise StructuralMapLocalizationError(
                "registered scans do not cover the complete query window"
            )
        points = np.concatenate([row.points_xyz_m for row in selected], axis=0)
        points = points[
            (points[:, 2] >= config.local_height_min_m)
            & (points[:, 2] <= config.local_height_max_m)
        ]
        query = vertical_persistence_xy(
            points,
            xy_resolution_m=config.query_voxel_resolution_m,
            minimum_vertical_span_m=config.minimum_vertical_span_m,
            minimum_height_bins=config.minimum_height_bins,
        )
        return query, {
            "start_source_time_ns": start_ns,
            "end_source_time_ns": end_source_time_ns,
            "scan_count": len(selected),
            "input_point_count": int(points.shape[0]),
            "query_point_count": int(query.shape[0]),
            "frame_id": "kiss_local",
        }

    def pose_at(self, source_time_ns: int, *, maximum_age_ns: int) -> TimedLocalPose:
        if not self.poses:
            raise StructuralMapLocalizationError("no local pose is available")
        times = np.asarray([row.source_time_ns for row in self.poses], dtype=np.int64)
        index = int(np.argmin(np.abs(times - source_time_ns)))
        row = self.poses[index]
        if abs(row.source_time_ns - source_time_ns) > maximum_age_ns:
            raise StructuralMapLocalizationError(
                "local pose is too far from map evidence time"
            )
        return row


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
        raise StructuralMapLocalizationError(
            "vertical persistence requires a nonempty Nx3 cloud"
        )
    points = points[np.all(np.isfinite(points), axis=1)]
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
        raise StructuralMapLocalizationError(
            "no XY columns pass the vertical-persistence gate"
        )
    xy_sum = np.zeros((group_count, 2), dtype=np.float64)
    xy_count = np.bincount(inverse, minlength=group_count)
    np.add.at(xy_sum, inverse, points[:, :2])
    return (xy_sum / xy_count[:, None])[admitted]


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
    # Repeated assignments are idempotent; sorting with np.unique is equivalent
    # but dominated initialization latency on the frozen 93-angle search.
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
    peak_index = np.asarray(
        np.unravel_index(int(np.argmax(correlation)), correlation.shape),
        dtype=np.int64,
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
    return rotation, target_mean - source_mean @ rotation


def directional_observability(
    map_xy: np.ndarray,
    transformed_query_xy: np.ndarray,
    target_indices: np.ndarray,
    admitted: np.ndarray,
    *,
    neighbors: int = 8,
) -> dict[str, object]:
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
        map_xy[target_indices[selected]],
        k=min(neighbors, map_xy.shape[0]),
        workers=-1,
    )
    neighborhood = map_xy[np.asarray(neighbor_index)]
    centered = neighborhood - np.mean(neighborhood, axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered)
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, :, 0]
    point = transformed_query_xy[selected]
    yaw_column = normal[:, 0] * (-point[:, 1]) + normal[:, 1] * point[:, 0]
    jacobian = np.column_stack((normal[:, 0], normal[:, 1], yaw_column))
    spectrum = np.linalg.eigvalsh(jacobian.T @ jacobian / selected.size)
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
    maximum_correspondence_m: float = 0.25,
    iterations: int = 30,
) -> dict[str, object]:
    tree = cKDTree(map_xy)
    total_rotation = np.asarray(rotation, dtype=np.float64)
    total_translation = np.asarray(translation_m, dtype=np.float64)
    previous_rmse = math.inf
    iteration = -1
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
    config: StructuralMapConfig | None = None,
) -> dict[str, object]:
    config = StructuralMapConfig() if config is None else config
    coarse = [
        correlate_at_yaw(
            map_xy,
            query_xy,
            angle_rad=math.radians(float(angle)),
            resolution_m=config.correlation_resolution_m,
        )
        for angle in np.arange(-180.0, 180.0, 5.0)
    ]
    coarse.sort(key=lambda item: float(item["overlap_fraction"]), reverse=True)
    best_angle_deg = math.degrees(float(coarse[0]["angle_rad"]))
    fine = [
        correlate_at_yaw(
            map_xy,
            query_xy,
            angle_rad=math.radians(float(angle)),
            resolution_m=config.correlation_resolution_m,
        )
        for angle in np.arange(best_angle_deg - 5.0, best_angle_deg + 5.25, 0.5)
    ]
    fine.sort(key=lambda item: float(item["overlap_fraction"]), reverse=True)
    best = fine[0]
    icp = refine_icp(
        map_xy,
        query_xy,
        np.asarray(best["rotation"]),
        np.asarray(best["translation_m"]),
        maximum_correspondence_m=config.global_maximum_correspondence_m,
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
        float(best["overlap_fraction"]) >= config.minimum_overlap_fraction
        and peak_ratio >= config.minimum_peak_ratio
        and float(icp["inlier_fraction"]) >= config.minimum_inlier_fraction
        and float(icp["rmse_m"]) <= config.maximum_rmse_m
        and float(icp["p95_m"]) <= config.maximum_global_p95_m
        and bool(icp["observability"]["observable"])
    )
    return {
        "rotation_matrix": np.asarray(icp["rotation"]).tolist(),
        "translation_m": np.asarray(icp["translation_m"]).tolist(),
        "yaw_deg": math.degrees(
            math.atan2(icp["rotation"][0, 1], icp["rotation"][0, 0])
        ),
        "correlation": {
            "best_overlap_fraction": float(best["overlap_fraction"]),
            "nonlocal_second_overlap_fraction": float(
                second_nonlocal["overlap_fraction"]
            ),
            "peak_ratio": peak_ratio,
        },
        "icp": {
            key: value
            for key, value in icp.items()
            if key not in {"rotation", "translation_m"}
        },
        "healthy": healthy,
    }


def _shortest_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


class AutomaticMapCorrectionEngine:
    """State machine for initialize-once position and heading-only updates."""

    def __init__(
        self,
        *,
        map_path: str | Path,
        map_key: str = "map_xy_all_5cm",
        map_version: int = 1,
        map_epoch: int = 1,
        config: StructuralMapConfig | None = None,
    ) -> None:
        config = StructuralMapConfig() if config is None else config
        path = Path(map_path)
        payload = path.read_bytes()
        with np.load(path, allow_pickle=False) as archive:
            if map_key not in archive.files:
                raise StructuralMapLocalizationError(f"map key {map_key!r} is missing")
            self.map_xy = np.asarray(archive[map_key], dtype=np.float64)
        if self.map_xy.ndim != 2 or self.map_xy.shape[1] != 2:
            raise StructuralMapLocalizationError(
                "structural map must contain Nx2 points"
            )
        if map_version < 1 or map_epoch < 1:
            raise StructuralMapLocalizationError(
                "map version and epoch must be positive"
            )
        self.map_digest = sha256(payload).digest()
        self.map_version = map_version
        self.map_epoch = map_epoch
        self.map_key = map_key
        self.config = config
        self.sequence = 0
        self.local_source_epoch: int | None = None
        self.rotation: np.ndarray | None = None
        self.translation_m: np.ndarray | None = None

    def bind_local_epoch(self, source_epoch: int) -> None:
        if source_epoch < 1:
            raise StructuralMapLocalizationError("local source epoch must be positive")
        if self.local_source_epoch != source_epoch:
            self.local_source_epoch = source_epoch
            self.sequence = 0
            self.rotation = None
            self.translation_m = None

    def _packet(
        self,
        *,
        rotation: np.ndarray,
        translation_m: np.ndarray,
        reference_time_ns: int,
        evidence_time_ns: int,
        application_time_ns: int,
        icp: dict[str, object],
    ) -> MapCorrectionPacketV1:
        if self.local_source_epoch is None:
            raise StructuralMapLocalizationError("local source epoch is not bound")
        age_ns = application_time_ns - evidence_time_ns
        if age_ns < 0 or age_ns > self.config.maximum_input_age_ns:
            raise StructuralMapLocalizationError(
                "map evidence is stale before publication"
            )
        observability = icp["observability"]
        self.sequence += 1
        variance = max(float(icp["rmse_m"]) ** 2, 1e-4)
        yaw_variance = max(math.radians(1.0) ** 2, variance / 4.0)
        return MapCorrectionPacketV1(
            sequence=self.sequence,
            map_epoch=self.map_epoch,
            local_source_epoch=self.local_source_epoch,
            map_version=self.map_version,
            reference_time_ns=reference_time_ns,
            evidence_time_ns=evidence_time_ns,
            application_time_ns=application_time_ns,
            publish_time_ns=application_time_ns,
            health_flags=(
                MapCorrectionHealth.ICP_CONVERGED
                | MapCorrectionHealth.INLIERS_VALID
                | MapCorrectionHealth.OBSERVABILITY_VALID
                | MapCorrectionHealth.MAP_IDENTITY_VALID
                | MapCorrectionHealth.CLOCK_VALID
            ),
            map_T_local=map_transform_from_row_se2(rotation, translation_m),
            fitness=float(icp["inlier_fraction"]),
            rmse_m=float(icp["rmse_m"]),
            min_eig=float(observability["minimum_eigenvalue"]),
            cond_number=float(observability["condition_number"]),
            covariance_diagonal=(variance, variance, 1e6, 1e6, 1e6, yaw_variance),
            map_digest=self.map_digest,
        )

    def initialize(
        self,
        query_xy: np.ndarray,
        *,
        reference_time_ns: int,
        evidence_time_ns: int,
        application_time_ns: int | None = None,
    ) -> MapCorrectionAttempt:
        start = time.perf_counter_ns()
        report = global_localize(self.map_xy, query_xy, config=self.config)
        runtime_ms = (time.perf_counter_ns() - start) * 1e-6
        now_ns = time.time_ns() if application_time_ns is None else application_time_ns
        reason = None if report["healthy"] else "global_geometry_gate"
        packet = None
        if reason is None:
            try:
                rotation = np.asarray(report["rotation_matrix"], dtype=np.float64)
                translation = np.asarray(report["translation_m"], dtype=np.float64)
                packet = self._packet(
                    rotation=rotation,
                    translation_m=translation,
                    reference_time_ns=reference_time_ns,
                    evidence_time_ns=evidence_time_ns,
                    application_time_ns=now_ns,
                    icp=report["icp"],
                )
                self.rotation = rotation
                self.translation_m = translation
            except StructuralMapLocalizationError as error:
                reason = str(error)
        return MapCorrectionAttempt(
            kind="global_initialization",
            accepted=packet is not None,
            rejection_reason=reason,
            reference_time_ns=reference_time_ns,
            evidence_time_ns=evidence_time_ns,
            runtime_ms=runtime_ms,
            report=report,
            packet=packet,
        )

    def track(
        self,
        query_xy: np.ndarray,
        *,
        local_position_xy_m: np.ndarray,
        reference_time_ns: int,
        evidence_time_ns: int,
        application_time_ns: int | None = None,
    ) -> MapCorrectionAttempt:
        if self.rotation is None or self.translation_m is None:
            raise StructuralMapLocalizationError(
                "global initialization is required before tracking"
            )
        start = time.perf_counter_ns()
        candidate = refine_icp(
            self.map_xy,
            query_xy,
            self.rotation,
            self.translation_m,
            maximum_correspondence_m=self.config.tracking_maximum_correspondence_m,
        )
        runtime_ms = (time.perf_counter_ns() - start) * 1e-6
        candidate_rotation = np.asarray(candidate["rotation"], dtype=np.float64)
        candidate_translation = np.asarray(candidate["translation_m"], dtype=np.float64)
        local_position = np.asarray(local_position_xy_m, dtype=np.float64)
        old_position = local_position @ self.rotation + self.translation_m
        candidate_position = local_position @ candidate_rotation + candidate_translation
        pose_jump_m = float(np.linalg.norm(candidate_position - old_position))
        old_yaw = math.degrees(math.atan2(self.rotation[0, 1], self.rotation[0, 0]))
        new_yaw = math.degrees(
            math.atan2(candidate_rotation[0, 1], candidate_rotation[0, 0])
        )
        yaw_jump_deg = abs(_shortest_degrees(new_yaw - old_yaw))
        observable = candidate["observability"]
        accepted = (
            float(candidate["inlier_fraction"]) >= self.config.minimum_inlier_fraction
            and float(candidate["rmse_m"]) <= self.config.maximum_rmse_m
            and float(candidate["p95_m"]) <= self.config.maximum_tracking_p95_m
            and bool(observable["observable"])
            and pose_jump_m <= self.config.maximum_pose_jump_m
            and yaw_jump_deg <= self.config.maximum_yaw_jump_deg
        )
        candidate["pose_jump_m"] = pose_jump_m
        candidate["yaw_jump_deg"] = yaw_jump_deg
        packet = None
        reason = None if accepted else "tracking_geometry_gate"
        if accepted:
            applied_translation = old_position - local_position @ candidate_rotation
            now_ns = (
                time.time_ns() if application_time_ns is None else application_time_ns
            )
            try:
                packet = self._packet(
                    rotation=candidate_rotation,
                    translation_m=applied_translation,
                    reference_time_ns=reference_time_ns,
                    evidence_time_ns=evidence_time_ns,
                    application_time_ns=now_ns,
                    icp=candidate,
                )
                self.rotation = candidate_rotation
                self.translation_m = applied_translation
            except StructuralMapLocalizationError as error:
                reason = str(error)
        report = {
            key: value
            for key, value in candidate.items()
            if key not in {"rotation", "translation_m"}
        }
        return MapCorrectionAttempt(
            kind="heading_only_tracking",
            accepted=packet is not None,
            rejection_reason=reason,
            reference_time_ns=reference_time_ns,
            evidence_time_ns=evidence_time_ns,
            runtime_ms=runtime_ms,
            report=report,
            packet=packet,
        )
