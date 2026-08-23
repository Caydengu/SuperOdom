"""Causal pelvis-heading recovery from an upside-down torso Livox IMU.

The G1-4123 Mid-360 is rigidly attached above the waist and has its sensor
z-axis opposite the torso z-axis.  Integrating the bias-corrected sensor z gyro
therefore estimates torso heading after a sign flip.  Removing the measured
waist-yaw excursion recovers pelvis heading without using Motive online.
"""

from __future__ import annotations

import numpy as np


class LivoxPelvisYawError(ValueError):
    """A yaw-recovery input violates the monotonic finite-data contract."""


def _validated_series(time_ns: np.ndarray, value: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time_ns, dtype=np.int64)
    data = np.asarray(value, dtype=np.float64)
    if time.ndim != 1 or data.shape != time.shape or time.size < 2:
        raise LivoxPelvisYawError(f"{name} must contain at least two scalar samples")
    if not np.all(np.diff(time) > 0):
        raise LivoxPelvisYawError(f"{name} timestamps must be strictly increasing")
    if not np.all(np.isfinite(data)):
        raise LivoxPelvisYawError(f"{name} values must be finite")
    return time, data


def recover_pelvis_yaw(
    *,
    output_time_ns: np.ndarray,
    imu_time_ns: np.ndarray,
    sensor_gyro_z_radps: np.ndarray,
    sensor_gyro_z_bias_radps: float,
    waist_time_ns: np.ndarray,
    waist_yaw_rad: np.ndarray,
    sensor_z_to_torso_yaw_sign: float = -1.0,
    compensate_waist: bool = True,
    maximum_imu_gap_ms: float = 25.0,
) -> np.ndarray:
    """Return relative pelvis yaw at ``output_time_ns``.

    Integration starts at the first IMU sample and is causal.  A gap larger
    than ``maximum_imu_gap_ms`` fails closed because interpolating angular rate
    across missing data would silently invent heading evidence.
    """

    imu_time, gyro_z = _validated_series(imu_time_ns, sensor_gyro_z_radps, "IMU")
    waist_time, waist = _validated_series(waist_time_ns, waist_yaw_rad, "waist")
    output = np.asarray(output_time_ns, dtype=np.int64)
    if output.ndim != 1 or output.size == 0 or not np.all(np.diff(output) > 0):
        raise LivoxPelvisYawError("output timestamps must be one strictly increasing series")
    if output[0] < imu_time[0] or output[-1] > imu_time[-1]:
        raise LivoxPelvisYawError("output timestamps exceed IMU coverage")
    if output[0] < waist_time[0] or output[-1] > waist_time[-1]:
        raise LivoxPelvisYawError("output timestamps exceed waist coverage")
    if not np.isfinite(sensor_gyro_z_bias_radps):
        raise LivoxPelvisYawError("gyro bias must be finite")
    if sensor_z_to_torso_yaw_sign not in (-1.0, 1.0):
        raise LivoxPelvisYawError("sensor z sign must be exactly -1 or +1")
    dt_s = np.diff(imu_time).astype(np.float64) * 1e-9
    if float(np.max(dt_s)) * 1e3 > maximum_imu_gap_ms:
        raise LivoxPelvisYawError("IMU gap exceeds the configured fail-closed limit")

    torso_rate = sensor_z_to_torso_yaw_sign * (
        gyro_z - float(sensor_gyro_z_bias_radps)
    )
    torso_yaw = np.zeros(imu_time.size, dtype=np.float64)
    torso_yaw[1:] = np.cumsum(0.5 * (torso_rate[:-1] + torso_rate[1:]) * dt_s)
    torso_output = np.interp(output, imu_time, torso_yaw)
    torso_output -= torso_output[0]
    if not compensate_waist:
        return torso_output

    waist_unwrapped = np.unwrap(waist)
    waist_output = np.interp(output, waist_time, waist_unwrapped)
    return torso_output - (waist_output - waist_output[0])


def yaw_quaternion_xyzw(yaw_rad: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw_rad, dtype=np.float64)
    if not np.all(np.isfinite(yaw)):
        raise LivoxPelvisYawError("yaw values must be finite")
    output = np.zeros(yaw.shape + (4,), dtype=np.float64)
    output[..., 2] = np.sin(0.5 * yaw)
    output[..., 3] = np.cos(0.5 * yaw)
    return output


def quaternion_yaw_z_up(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise LivoxPelvisYawError("quaternions must have shape (N, 4)")
    norm = np.linalg.norm(quaternion, axis=1)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0.0):
        raise LivoxPelvisYawError("quaternions must be finite and nonzero")
    x, y, z, w = (quaternion[:, index] / norm for index in range(4))
    return np.unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def rehead_planar_trajectory(
    position_xyz_m: np.ndarray,
    source_yaw_rad: np.ndarray,
    replacement_yaw_rad: np.ndarray,
) -> np.ndarray:
    """Reintegrate source-local planar increments under a replacement heading."""

    position = np.asarray(position_xyz_m, dtype=np.float64)
    source_yaw = np.asarray(source_yaw_rad, dtype=np.float64)
    replacement_yaw = np.asarray(replacement_yaw_rad, dtype=np.float64)
    if position.ndim != 2 or position.shape[1] != 3 or position.shape[0] < 2:
        raise LivoxPelvisYawError("positions must have shape (N, 3), N >= 2")
    if source_yaw.shape != (position.shape[0],) or replacement_yaw.shape != source_yaw.shape:
        raise LivoxPelvisYawError("source and replacement yaw must match position count")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(source_yaw)) or not np.all(np.isfinite(replacement_yaw)):
        raise LivoxPelvisYawError("reheading inputs must be finite")

    delta_world = np.diff(position[:, :2], axis=0)
    c_source = np.cos(source_yaw[:-1])
    s_source = np.sin(source_yaw[:-1])
    delta_local = np.column_stack(
        (
            c_source * delta_world[:, 0] + s_source * delta_world[:, 1],
            -s_source * delta_world[:, 0] + c_source * delta_world[:, 1],
        )
    )
    c_replacement = np.cos(replacement_yaw[:-1])
    s_replacement = np.sin(replacement_yaw[:-1])
    delta_reheaded = np.column_stack(
        (
            c_replacement * delta_local[:, 0] - s_replacement * delta_local[:, 1],
            s_replacement * delta_local[:, 0] + c_replacement * delta_local[:, 1],
        )
    )
    output = position.copy()
    output[0, :2] = position[0, :2]
    output[1:, :2] = output[0, :2] + np.cumsum(delta_reheaded, axis=0)
    return output
