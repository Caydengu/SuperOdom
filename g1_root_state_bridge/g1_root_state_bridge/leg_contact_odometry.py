"""Minimal G1 leg forward kinematics and causal stance-foot odometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


class LegContactOdometryError(ValueError):
    """The kinematic model or contact sequence violates its contract."""


def _rpy_rotation(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=np.float64)
    ry = np.asarray(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=np.float64)
    rz = np.asarray(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float64)
    return rz @ ry @ rx


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    unit = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(unit))
    if not np.isfinite(norm) or norm <= 0.0:
        raise LegContactOdometryError("joint axis must be finite and nonzero")
    x, y, z = unit / norm
    c, s, v = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.asarray(
        (
            (x * x * v + c, x * y * v - z * s, x * z * v + y * s),
            (y * x * v + z * s, y * y * v + c, y * z * v - x * s),
            (z * x * v - y * s, z * y * v + x * s, z * z * v + c),
        ),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class _Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


class UrdfLegKinematics:
    """Evaluate pelvis-to-foot transforms from an ordinary serial-tree URDF."""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        root_link: str = "pelvis",
        left_foot_link: str = "left_ankle_roll_link",
        right_foot_link: str = "right_ankle_roll_link",
        sole_center_xyz_m: tuple[float, float, float] = (0.035, 0.0, -0.03),
    ):
        path = Path(urdf_path)
        if not path.is_file():
            raise LegContactOdometryError(f"URDF is missing: {path}")
        root = ET.parse(path).getroot()
        by_child: dict[str, _Joint] = {}
        for node in root.findall("joint"):
            parent = node.find("parent")
            child = node.find("child")
            if parent is None or child is None:
                raise LegContactOdometryError("URDF joint is missing parent or child")
            origin = node.find("origin")
            axis = node.find("axis")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            axis_xyz = np.fromstring(axis.get("xyz", "0 0 1") if axis is not None else "0 0 1", sep=" ")
            joint = _Joint(
                name=str(node.get("name")),
                joint_type=str(node.get("type", "fixed")),
                parent=str(parent.get("link")),
                child=str(child.get("link")),
                xyz=xyz,
                rpy=rpy,
                axis=axis_xyz,
            )
            if joint.child in by_child:
                raise LegContactOdometryError(f"multiple parents for {joint.child}")
            by_child[joint.child] = joint
        self._root_link = root_link
        self._chains = (
            self._chain(by_child, left_foot_link),
            self._chain(by_child, right_foot_link),
        )
        self._sole_center = np.asarray((*sole_center_xyz_m, 1.0), dtype=np.float64)
        self.required_joint_names = tuple(
            dict.fromkeys(
                joint.name
                for chain in self._chains
                for joint in chain
                if joint.joint_type != "fixed"
            )
        )

    def _chain(self, by_child: dict[str, _Joint], leaf: str) -> tuple[_Joint, ...]:
        reverse: list[_Joint] = []
        link = leaf
        while link != self._root_link:
            if link not in by_child:
                raise LegContactOdometryError(f"no chain from {self._root_link} to {leaf}")
            joint = by_child[link]
            reverse.append(joint)
            link = joint.parent
        return tuple(reversed(reverse))

    @staticmethod
    def _joint_transform(joint: _Joint, angle: float) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = _rpy_rotation(joint.rpy)
        transform[:3, 3] = joint.xyz
        if joint.joint_type in ("revolute", "continuous"):
            rotation = np.eye(4, dtype=np.float64)
            rotation[:3, :3] = _axis_rotation(joint.axis, angle)
            transform = transform @ rotation
        elif joint.joint_type != "fixed":
            raise LegContactOdometryError(f"unsupported joint type {joint.joint_type}")
        return transform

    def foot_positions(self, joint_position_by_name: dict[str, float]) -> np.ndarray:
        output = []
        for chain in self._chains:
            transform = np.eye(4, dtype=np.float64)
            for joint in chain:
                angle = float(joint_position_by_name[joint.name]) if joint.joint_type != "fixed" else 0.0
                transform = transform @ self._joint_transform(joint, angle)
            output.append((transform @ self._sole_center)[:3])
        return np.asarray(output, dtype=np.float64)


def stance_foot_odometry(
    foot_position_pelvis_m: np.ndarray,
    pelvis_yaw_rad: np.ndarray,
    *,
    switch_height_margin_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover pelvis translation by holding the lower stance foot fixed."""

    feet = np.asarray(foot_position_pelvis_m, dtype=np.float64)
    yaw = np.asarray(pelvis_yaw_rad, dtype=np.float64)
    if feet.ndim != 3 or feet.shape[1:] != (2, 3) or feet.shape[0] < 2:
        raise LegContactOdometryError("foot positions must have shape (N, 2, 3)")
    if yaw.shape != (feet.shape[0],) or not np.all(np.isfinite(feet)) or not np.all(np.isfinite(yaw)):
        raise LegContactOdometryError("foot positions and yaw must be finite and length matched")
    if not np.isfinite(switch_height_margin_m) or switch_height_margin_m < 0.0:
        raise LegContactOdometryError("switch height margin must be finite and nonnegative")

    position = np.zeros((feet.shape[0], 3), dtype=np.float64)
    stance = np.empty(feet.shape[0], dtype=np.int8)
    current = int(np.argmin(feet[0, :, 2]))
    c0, s0 = np.cos(yaw[0]), np.sin(yaw[0])
    r0 = np.asarray(((c0, -s0, 0), (s0, c0, 0), (0, 0, 1)), dtype=np.float64)
    anchor = r0 @ feet[0, current]
    stance[0] = current
    for index in range(1, feet.shape[0]):
        other = 1 - current
        if feet[index, other, 2] < feet[index, current, 2] - switch_height_margin_m:
            current = other
            c, s = np.cos(yaw[index]), np.sin(yaw[index])
            rotation = np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float64)
            anchor = position[index - 1] + rotation @ feet[index, current]
        c, s = np.cos(yaw[index]), np.sin(yaw[index])
        rotation = np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float64)
        position[index] = anchor - rotation @ feet[index, current]
        stance[index] = current
    return position, stance
