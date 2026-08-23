"""Pinned KISS-ICP registration backend for the selected G1 local lane."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import time

import numpy as np


class KissRegistrationError(RuntimeError):
    """The KISS runtime or one registration result is not admissible."""


@dataclass(frozen=True)
class RegistrationResult:
    sensor0_T_sensor: np.ndarray
    runtime_ms: float
    adaptive_threshold: float


class KissRegistration:
    """Small stateful wrapper that pins the exact research configuration."""

    def __init__(
        self,
        *,
        voxel_size_m: float = 0.15,
        minimum_range_m: float = 0.5,
        maximum_range_m: float = 15.0,
        required_version: str = "1.3.0",
    ) -> None:
        installed = importlib.metadata.version("kiss-icp")
        if installed != required_version:
            raise KissRegistrationError(
                f"kiss-icp version must be {required_version}, found {installed}"
            )
        self._settings = (voxel_size_m, minimum_range_m, maximum_range_m)
        self._create_odometry()

    def _create_odometry(self) -> None:
        from kiss_icp.config import KISSConfig
        from kiss_icp.kiss_icp import KissICP

        voxel_size_m, minimum_range_m, maximum_range_m = self._settings
        config = KISSConfig()
        config.data.min_range = minimum_range_m
        config.data.max_range = maximum_range_m
        config.data.deskew = False
        config.mapping.voxel_size = voxel_size_m
        self._odometry = KissICP(config)

    def reset(self) -> None:
        """Discard the local map after a source-clock epoch change."""
        self._create_odometry()

    def register(self, points_xyz_m: np.ndarray) -> RegistrationResult:
        points = np.asarray(points_xyz_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 10:
            raise KissRegistrationError("registration requires at least ten 3D points")
        if not np.all(np.isfinite(points)):
            raise KissRegistrationError("registration points must be finite")
        start_ns = time.perf_counter_ns()
        self._odometry.register_frame(points, np.zeros(points.shape[0], dtype=np.float64))
        runtime_ms = (time.perf_counter_ns() - start_ns) * 1e-6
        pose = np.asarray(self._odometry.last_pose, dtype=np.float64).copy()
        threshold = float(self._odometry.adaptive_threshold.get_threshold())
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)) or not np.isfinite(threshold):
            raise KissRegistrationError("KISS returned a non-finite result")
        return RegistrationResult(pose, runtime_ms, threshold)
