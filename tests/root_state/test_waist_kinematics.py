import numpy as np

from g1_root_state_bridge.waist_kinematics import (
    invert_transform,
    pelvis_T_mid360,
    sensor_local_pose_to_pelvis_local_pose,
)


def test_zero_waist_chain_matches_frozen_urdf_translation() -> None:
    transform = pelvis_T_mid360(0.0, 0.0, 0.0)
    np.testing.assert_allclose(transform[:3, 3], (-0.00368, 0.00003, 0.46018), atol=1e-12)
    np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12)


def test_sensor_motion_from_waist_articulation_recovers_stationary_pelvis() -> None:
    initial = pelvis_T_mid360(0.0, 0.0, 0.0)
    current = pelvis_T_mid360(0.35, -0.10, 0.16)
    sensor0_T_sensor_t = invert_transform(initial) @ current
    recovered = sensor_local_pose_to_pelvis_local_pose(
        sensor0_T_sensor_t,
        pelvis0_T_sensor0=initial,
        pelvis_t_T_sensor_t=current,
    )
    np.testing.assert_allclose(recovered, np.eye(4), atol=1e-12)
