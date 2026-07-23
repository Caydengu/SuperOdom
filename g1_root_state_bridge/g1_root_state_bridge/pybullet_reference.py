"""Independent PyBullet pose fixture for the pelvis-to-Mid-360 chain.

This module is validation-only.  Production reconstruction lives in
``kinematics.py`` and never imports or calls PyBullet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from g1_root_state_bridge.joint_contract import TimedJointSample
from g1_root_state_bridge.kinematics import (
    KINEMATICS_URDF_PATH,
    KinematicsContractError,
    PHYSICAL_SENSOR_FRAME,
    WAIST_JOINT_NAMES,
    _invert_transform,
    _validated_transform,
)


class PyBulletPelvisReference:
    """Load the same URDF through an independent kinematic implementation."""

    def __init__(self, urdf_path: str | Path = KINEMATICS_URDF_PATH):
        try:
            import pybullet as pybullet
        except ImportError as error:  # pragma: no cover - dependency gate
            raise RuntimeError("PyBullet is required for the reference fixture") from error
        self._pybullet: Any = pybullet
        self._client = pybullet.connect(pybullet.DIRECT)
        if self._client < 0:
            raise RuntimeError("PyBullet DIRECT connection failed")
        self.urdf_path = Path(urdf_path).resolve()
        self._robot = pybullet.loadURDF(
            str(self.urdf_path), useFixedBase=True, physicsClientId=self._client
        )
        self._joint_by_name: dict[str, int] = {}
        self._sensor_link_id: int | None = None
        for index in range(
            pybullet.getNumJoints(self._robot, physicsClientId=self._client)
        ):
            info = pybullet.getJointInfo(
                self._robot, index, physicsClientId=self._client
            )
            self._joint_by_name[info[1].decode("utf-8")] = index
            if info[12].decode("utf-8") == PHYSICAL_SENSOR_FRAME:
                self._sensor_link_id = index
        missing = sorted(set(WAIST_JOINT_NAMES) - set(self._joint_by_name))
        if missing or self._sensor_link_id is None:
            self.close()
            raise KinematicsContractError(
                f"PyBullet URDF contract mismatch; missing_joints={missing}, "
                f"sensor_link={self._sensor_link_id}"
            )

    def close(self) -> None:
        if getattr(self, "_client", -1) >= 0:
            self._pybullet.disconnect(self._client)
            self._client = -1

    def pelvis_T_observed(
        self,
        joint_sample: TimedJointSample,
        physical_sensor_T_observed: np.ndarray,
    ) -> np.ndarray:
        physical_sensor_T_observed = _validated_transform(
            physical_sensor_T_observed, "physical_sensor_T_observed"
        )
        sample = joint_sample.canonicalized()
        position_by_name = dict(zip(sample.names, sample.position, strict=True))
        for name in WAIST_JOINT_NAMES:
            self._pybullet.resetJointState(
                self._robot,
                self._joint_by_name[name],
                position_by_name[name],
                physicsClientId=self._client,
            )
        state = self._pybullet.getLinkState(
            self._robot,
            self._sensor_link_id,
            computeForwardKinematics=True,
            physicsClientId=self._client,
        )
        pelvis_T_sensor = np.eye(4)
        pelvis_T_sensor[:3, :3] = np.asarray(
            self._pybullet.getMatrixFromQuaternion(state[5])
        ).reshape(3, 3)
        pelvis_T_sensor[:3, 3] = np.asarray(state[4])
        return pelvis_T_sensor @ physical_sensor_T_observed

    def recover_pelvis_pose(
        self,
        world_T_observed: np.ndarray,
        joint_sample: TimedJointSample,
        physical_sensor_T_observed: np.ndarray,
    ) -> np.ndarray:
        world_T_observed = _validated_transform(world_T_observed, "world_T_observed")
        return world_T_observed @ _invert_transform(
            self.pelvis_T_observed(joint_sample, physical_sensor_T_observed)
        )
