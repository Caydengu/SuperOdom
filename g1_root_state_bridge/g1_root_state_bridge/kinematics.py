"""Pinocchio recovery of G1 pelvis pose and twist from an observed head frame.

Transform names use ``A_T_B`` to mean a transform that maps coordinates in B
into A.  The URDF contains the *physical* upside-down Mid-360 frame.  Any
virtual frame published by an estimator must therefore arrive as a separate
``physical_sensor_T_observed`` calibration; it is never hidden in a second
URDF variant.

All velocities returned by this module are expressed in the world frame and
refer to the named frame origin.  This deliberately differs from ROS
``nav_msgs/Odometry``'s conventional child-frame twist; the ROS adapter owns
that conversion before it invokes this pure kinematic core.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from g1_root_state_bridge.joint_contract import TimedJointSample


KINEMATICS_URDF_PATH = (
    Path(__file__).resolve().parent / "assets" / "g1_pelvis_mid360_kinematics.urdf"
)
WAIST_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
PHYSICAL_SENSOR_FRAME = "mid360_link"


class KinematicsContractError(ValueError):
    """A transform, model, or joint sample violates the frozen FK contract."""


@dataclass(frozen=True)
class PelvisState:
    world_T_pelvis: np.ndarray
    linear_velocity_world: np.ndarray
    angular_velocity_world: np.ndarray


def _validated_transform(value: np.ndarray, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4):
        raise KinematicsContractError(f"{name} must have shape (4, 4)")
    if not np.all(np.isfinite(transform)):
        raise KinematicsContractError(f"{name} must be finite")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-10):
        raise KinematicsContractError(f"{name} must be a homogeneous transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise KinematicsContractError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
        raise KinematicsContractError(f"{name} rotation must be proper")
    return transform


def _validated_vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise KinematicsContractError(f"{name} must be one finite 3-vector")
    return vector


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ transform[:3, 3])
    return inverse


class PelvisKinematics:
    """Compute pelvis state with a fixed-base Pinocchio waist model."""

    def __init__(self, urdf_path: str | Path = KINEMATICS_URDF_PATH):
        try:
            import pinocchio as pin
        except ImportError as error:  # pragma: no cover - dependency gate
            raise RuntimeError("Pinocchio is required for PelvisKinematics") from error

        self._pin: Any = pin
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.is_file():
            raise KinematicsContractError(f"kinematics URDF is missing: {self.urdf_path}")
        self._model = pin.buildModelFromUrdf(str(self.urdf_path))
        self._data = self._model.createData()
        self._sensor_frame_id = self._model.getFrameId(PHYSICAL_SENSOR_FRAME)
        if self._sensor_frame_id >= self._model.nframes:
            raise KinematicsContractError(
                f"URDF has no {PHYSICAL_SENSOR_FRAME!r} frame"
            )
        self._joint_ids: dict[str, int] = {}
        for name in WAIST_JOINT_NAMES:
            joint_id = self._model.getJointId(name)
            if joint_id == 0:
                raise KinematicsContractError(f"URDF has no {name!r} joint")
            joint = self._model.joints[joint_id]
            if joint.nq != 1 or joint.nv != 1:
                raise KinematicsContractError(f"{name!r} is not one revolute DOF")
            self._joint_ids[name] = joint_id

    def _configuration(
        self, joint_sample: TimedJointSample
    ) -> tuple[np.ndarray, np.ndarray]:
        sample = joint_sample.canonicalized()
        by_name_q = dict(zip(sample.names, sample.position, strict=True))
        by_name_dq = dict(zip(sample.names, sample.velocity, strict=True))
        q = np.zeros(self._model.nq, dtype=np.float64)
        dq = np.zeros(self._model.nv, dtype=np.float64)
        for name, joint_id in self._joint_ids.items():
            joint = self._model.joints[joint_id]
            q[joint.idx_q] = by_name_q[name]
            dq[joint.idx_v] = by_name_dq[name]
        return q, dq

    def _physical_sensor_kinematics(
        self, joint_sample: TimedJointSample
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q, dq = self._configuration(joint_sample)
        self._pin.forwardKinematics(self._model, self._data, q, dq)
        self._pin.updateFramePlacements(self._model, self._data)
        placement = self._data.oMf[self._sensor_frame_id]
        velocity = self._pin.getFrameVelocity(
            self._model,
            self._data,
            self._sensor_frame_id,
            self._pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        pelvis_T_sensor = np.eye(4)
        pelvis_T_sensor[:3, :3] = np.asarray(placement.rotation)
        pelvis_T_sensor[:3, 3] = np.asarray(placement.translation).reshape(3)
        relative_linear_pelvis = np.asarray(velocity.linear).reshape(3)
        relative_angular_pelvis = np.asarray(velocity.angular).reshape(3)
        return pelvis_T_sensor, relative_linear_pelvis, relative_angular_pelvis

    def pelvis_T_observed(
        self,
        joint_sample: TimedJointSample,
        physical_sensor_T_observed: np.ndarray,
    ) -> np.ndarray:
        physical_sensor_T_observed = _validated_transform(
            physical_sensor_T_observed, "physical_sensor_T_observed"
        )
        pelvis_T_sensor, _, _ = self._physical_sensor_kinematics(joint_sample)
        return pelvis_T_sensor @ physical_sensor_T_observed

    def recover_pelvis_pose(
        self,
        world_T_observed: np.ndarray,
        joint_sample: TimedJointSample,
        physical_sensor_T_observed: np.ndarray,
    ) -> np.ndarray:
        world_T_observed = _validated_transform(world_T_observed, "world_T_observed")
        pelvis_T_observed = self.pelvis_T_observed(
            joint_sample, physical_sensor_T_observed
        )
        return world_T_observed @ _invert_transform(pelvis_T_observed)

    def compute(
        self,
        *,
        world_T_observed: np.ndarray,
        observed_linear_velocity_world: np.ndarray,
        observed_angular_velocity_world: np.ndarray,
        joint_sample: TimedJointSample,
        physical_sensor_T_observed: np.ndarray,
    ) -> PelvisState:
        world_T_observed = _validated_transform(world_T_observed, "world_T_observed")
        physical_sensor_T_observed = _validated_transform(
            physical_sensor_T_observed, "physical_sensor_T_observed"
        )
        observed_linear_velocity_world = _validated_vector(
            observed_linear_velocity_world, "observed_linear_velocity_world"
        )
        observed_angular_velocity_world = _validated_vector(
            observed_angular_velocity_world, "observed_angular_velocity_world"
        )

        pelvis_T_sensor, sensor_linear_pelvis, sensor_angular_pelvis = (
            self._physical_sensor_kinematics(joint_sample)
        )
        pelvis_T_observed = pelvis_T_sensor @ physical_sensor_T_observed
        world_T_pelvis = world_T_observed @ _invert_transform(pelvis_T_observed)

        # The observed frame can have a fixed offset from the physical sensor.
        # Its relative origin velocity gets the usual omega cross r term.
        sensor_to_observed_pelvis = (
            pelvis_T_sensor[:3, :3] @ physical_sensor_T_observed[:3, 3]
        )
        observed_relative_linear_pelvis = sensor_linear_pelvis + np.cross(
            sensor_angular_pelvis, sensor_to_observed_pelvis
        )
        observed_relative_angular_pelvis = sensor_angular_pelvis

        pelvis_rotation_world = world_T_pelvis[:3, :3]
        pelvis_to_observed_world = (
            pelvis_rotation_world @ pelvis_T_observed[:3, 3]
        )
        pelvis_angular_world = observed_angular_velocity_world - (
            pelvis_rotation_world @ observed_relative_angular_pelvis
        )
        pelvis_linear_world = (
            observed_linear_velocity_world
            - np.cross(pelvis_angular_world, pelvis_to_observed_world)
            - pelvis_rotation_world @ observed_relative_linear_pelvis
        )
        return PelvisState(
            world_T_pelvis=world_T_pelvis,
            linear_velocity_world=pelvis_linear_world,
            angular_velocity_world=pelvis_angular_world,
        )
