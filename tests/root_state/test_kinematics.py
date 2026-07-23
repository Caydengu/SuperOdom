from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from g1_root_state_bridge.joint_contract import (
    CANONICAL_G1_JOINT_NAMES,
    TimedJointSample,
)
from g1_root_state_bridge.kinematics import (
    KINEMATICS_URDF_PATH,
    PelvisKinematics,
)
from g1_root_state_bridge.pybullet_reference import PyBulletPelvisReference


WAIST_INDEX = {
    name: CANONICAL_G1_JOINT_NAMES.index(name)
    for name in (
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    )
}


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))


def _transform(rotation: np.ndarray, translation: tuple[float, float, float]) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def _so3_exp(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-14:
        return np.eye(3)
    axis = vector / angle
    skew = np.array(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        )
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _so3_log(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-10:
        return np.array(
            (
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            )
        ) / 2.0
    return angle / (2.0 * math.sin(angle)) * np.array(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    )


def _orientation_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(_so3_log(left @ right.T)))


def _sample(
    waist_position: tuple[float, float, float],
    waist_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> TimedJointSample:
    q = [0.0] * len(CANONICAL_G1_JOINT_NAMES)
    dq = [0.0] * len(CANONICAL_G1_JOINT_NAMES)
    for name, value, velocity in zip(
        WAIST_INDEX, waist_position, waist_velocity, strict=True
    ):
        q[WAIST_INDEX[name]] = value
        dq[WAIST_INDEX[name]] = velocity
    return TimedJointSample(
        stamp_ns=1_000_000_000,
        receipt_ns=1_000_500_000,
        names=CANONICAL_G1_JOINT_NAMES,
        position=tuple(q),
        velocity=tuple(dq),
        sequence=10,
        source_epoch=3,
    )


@pytest.fixture(scope="module")
def implementations() -> tuple[PelvisKinematics, PyBulletPelvisReference]:
    assert KINEMATICS_URDF_PATH.is_file()
    pinocchio = PelvisKinematics(KINEMATICS_URDF_PATH)
    pybullet = PyBulletPelvisReference(KINEMATICS_URDF_PATH)
    yield pinocchio, pybullet
    pybullet.close()


def test_fixture_is_hash_bound_and_contains_only_the_required_chain() -> None:
    expected_source_sha = "c34d7ebbd1e6345bb815a54e05523086a68f4105d2ed414be6ca50f52f08473f"
    text = Path(KINEMATICS_URDF_PATH).read_text(encoding="utf-8")
    assert f"source_full_urdf_sha256={expected_source_sha}" in text
    for name in (*WAIST_INDEX, "pelvis", "mid360_link"):
        assert name in text
    for excluded in ("hip_joint", "shoulder_joint", "mesh filename"):
        assert excluded not in text


@pytest.mark.parametrize(
    "waist_position",
    [
        (0.0, 0.0, 0.0),
        (0.43, -0.21, 0.17),
        (-0.81, 0.31, -0.27),
    ],
)
def test_pinocchio_and_pybullet_pose_agree_for_named_waist_fixtures(
    implementations: tuple[PelvisKinematics, PyBulletPelvisReference],
    waist_position: tuple[float, float, float],
) -> None:
    pinocchio, pybullet = implementations
    joints = _sample(waist_position)
    # Exercise a nonidentity observed-frame calibration even though the current
    # SuperOdometry bridge is expected to use the physical Mid-360 frame.
    physical_sensor_T_observed = _transform(
        _rot_x(math.pi) @ _rot_y(0.037),
        (0.0, 0.0, 0.0),
    )

    pin_transform = pinocchio.pelvis_T_observed(
        joints, physical_sensor_T_observed
    )
    bullet_transform = pybullet.pelvis_T_observed(
        joints, physical_sensor_T_observed
    )

    assert np.linalg.norm(pin_transform[:3, 3] - bullet_transform[:3, 3]) <= 1e-4
    assert _orientation_error(pin_transform[:3, :3], bullet_transform[:3, :3]) <= 1e-4

    world_T_observed = _transform(
        _rot_z(0.72) @ _rot_y(-0.13) @ _rot_x(0.09),
        (1.1, -0.7, 1.42),
    )
    pin_world_T_pelvis = pinocchio.recover_pelvis_pose(
        world_T_observed, joints, physical_sensor_T_observed
    )
    bullet_world_T_pelvis = pybullet.recover_pelvis_pose(
        world_T_observed, joints, physical_sensor_T_observed
    )
    assert np.linalg.norm(
        pin_world_T_pelvis[:3, 3] - bullet_world_T_pelvis[:3, 3]
    ) <= 1e-4
    assert _orientation_error(
        pin_world_T_pelvis[:3, :3], bullet_world_T_pelvis[:3, :3]
    ) <= 1e-4


def test_pinocchio_matches_deterministic_randomized_pybullet_fixtures(
    implementations: tuple[PelvisKinematics, PyBulletPelvisReference],
) -> None:
    pinocchio, pybullet = implementations
    generator = np.random.default_rng(20260722)
    limits = np.array(((-2.618, 2.618), (-0.52, 0.52), (-0.52, 0.52)))
    physical_sensor_T_observed = _transform(
        _rot_z(-0.03) @ _rot_x(math.pi - 0.02),
        (0.004, -0.003, 0.002),
    )
    maximum_translation_error = 0.0
    maximum_orientation_error = 0.0
    for _ in range(100):
        waist = tuple(generator.uniform(limits[:, 0], limits[:, 1]))
        joints = _sample(waist)
        pin_transform = pinocchio.pelvis_T_observed(
            joints, physical_sensor_T_observed
        )
        bullet_transform = pybullet.pelvis_T_observed(
            joints, physical_sensor_T_observed
        )
        maximum_translation_error = max(
            maximum_translation_error,
            float(np.linalg.norm(pin_transform[:3, 3] - bullet_transform[:3, 3])),
        )
        maximum_orientation_error = max(
            maximum_orientation_error,
            _orientation_error(pin_transform[:3, :3], bullet_transform[:3, :3]),
        )
    assert maximum_translation_error <= 1e-4
    assert maximum_orientation_error <= 1e-4


def test_frame_jacobian_twist_recovers_moving_pelvis_with_waist_motion(
    implementations: tuple[PelvisKinematics, PyBulletPelvisReference],
) -> None:
    pinocchio, pybullet = implementations
    waist = np.array((0.31, -0.18, 0.22))
    waist_velocity = np.array((0.83, -0.47, 0.59))
    pelvis_linear_world = np.array((0.74, -0.29, 0.18))
    pelvis_angular_world = np.array((0.31, -0.26, 0.57))
    world_T_pelvis = _transform(
        _rot_z(0.62) @ _rot_y(-0.19) @ _rot_x(0.11),
        (0.8, -1.2, 0.91),
    )
    physical_sensor_T_observed = _transform(
        _rot_x(math.pi) @ _rot_y(0.041),
        (0.006, -0.004, 0.003),
    )
    # PyBullet exposes link poses through single-precision values.  A 1 ms
    # central-difference interval stays small relative to the motion while
    # avoiding the round-off amplification seen at 10 us.
    dt = 1e-3

    observed_poses = []
    for sign in (-1.0, 0.0, 1.0):
        moved_pelvis = np.array(world_T_pelvis, copy=True)
        moved_pelvis[:3, :3] = (
            _so3_exp(sign * dt * pelvis_angular_world) @ world_T_pelvis[:3, :3]
        )
        moved_pelvis[:3, 3] = (
            world_T_pelvis[:3, 3] + sign * dt * pelvis_linear_world
        )
        moved_joints = _sample(
            tuple(waist + sign * dt * waist_velocity),
            tuple(waist_velocity),
        )
        observed_poses.append(
            moved_pelvis
            @ pybullet.pelvis_T_observed(
                moved_joints, physical_sensor_T_observed
            )
        )

    minus, current, plus = observed_poses
    observed_linear_world = (plus[:3, 3] - minus[:3, 3]) / (2.0 * dt)
    observed_angular_world = _so3_log(plus[:3, :3] @ minus[:3, :3].T) / (
        2.0 * dt
    )
    result = pinocchio.compute(
        world_T_observed=current,
        observed_linear_velocity_world=observed_linear_world,
        observed_angular_velocity_world=observed_angular_world,
        joint_sample=_sample(tuple(waist), tuple(waist_velocity)),
        physical_sensor_T_observed=physical_sensor_T_observed,
    )

    assert np.linalg.norm(result.world_T_pelvis[:3, 3] - world_T_pelvis[:3, 3]) <= 1e-4
    assert _orientation_error(
        result.world_T_pelvis[:3, :3], world_T_pelvis[:3, :3]
    ) <= 1e-4
    assert np.linalg.norm(result.linear_velocity_world - pelvis_linear_world) <= 2e-4
    assert np.linalg.norm(result.angular_velocity_world - pelvis_angular_world) <= 2e-4
