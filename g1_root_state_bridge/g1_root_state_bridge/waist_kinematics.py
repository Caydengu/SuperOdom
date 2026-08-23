"""Dependency-free G1 pelvis-to-Mid360 waist kinematics.

This is the exact four-joint chain frozen in
``assets/g1_pelvis_mid360_kinematics.urdf``.  It exists so offline odometry
comparators can obey the pelvis output contract without installing Pinocchio.
"""

from __future__ import annotations

import numpy as np


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def invert_transform(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError("transform must have shape (4,4)")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -(value[:3, :3].T @ value[:3, 3])
    return result


def pelvis_T_mid360(waist_yaw: float, waist_roll: float, waist_pitch: float) -> np.ndarray:
    """Return the physical upside-down Mid360 pose in pelvis coordinates."""
    angles = np.asarray((waist_yaw, waist_roll, waist_pitch), dtype=np.float64)
    if not np.all(np.isfinite(angles)):
        raise ValueError("waist angles must be finite")
    pelvis_T_yaw = _transform(_rotation_z(float(waist_yaw)), np.zeros(3))
    yaw_T_roll = _transform(
        _rotation_x(float(waist_roll)), np.asarray((-0.0039635, 0.0, 0.044))
    )
    roll_T_torso = _transform(_rotation_y(float(waist_pitch)), np.zeros(3))
    # URDF fixed-joint rpy convention is Rz(yaw) @ Ry(pitch) @ Rx(roll).
    torso_T_sensor = _transform(
        _rotation_y(0.04014257279586953) @ _rotation_x(np.pi),
        np.asarray((0.0002835, 0.00003, 0.41618)),
    )
    return pelvis_T_yaw @ yaw_T_roll @ roll_T_torso @ torso_T_sensor


def sensor_local_pose_to_pelvis_local_pose(
    sensor0_T_sensor_t: np.ndarray,
    *,
    pelvis0_T_sensor0: np.ndarray,
    pelvis_t_T_sensor_t: np.ndarray,
) -> np.ndarray:
    """Convert sensor odometry into the initial-pelvis local frame."""
    return (
        np.asarray(pelvis0_T_sensor0, dtype=np.float64)
        @ np.asarray(sensor0_T_sensor_t, dtype=np.float64)
        @ invert_transform(np.asarray(pelvis_t_T_sensor_t, dtype=np.float64))
    )
