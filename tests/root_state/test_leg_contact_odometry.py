from pathlib import Path

import numpy as np

from g1_root_state_bridge.leg_contact_odometry import UrdfLegKinematics, stance_foot_odometry


def test_urdf_chain_applies_revolute_joint(tmp_path: Path):
    urdf = tmp_path / "leg.urdf"
    urdf.write_text(
        """<robot name='test'>
        <link name='pelvis'/><link name='left_ankle_roll_link'/><link name='right_ankle_roll_link'/>
        <joint name='left_joint' type='revolute'><parent link='pelvis'/><child link='left_ankle_roll_link'/><origin xyz='1 0 0'/><axis xyz='0 0 1'/></joint>
        <joint name='right_joint' type='fixed'><parent link='pelvis'/><child link='right_ankle_roll_link'/><origin xyz='0 -1 0'/></joint>
        </robot>""",
        encoding="utf-8",
    )
    model = UrdfLegKinematics(urdf, sole_center_xyz_m=(1.0, 0.0, 0.0))
    feet = model.foot_positions({"left_joint": np.pi / 2.0})
    np.testing.assert_allclose(feet[0], [1.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(feet[1], [1.0, -1.0, 0.0], atol=1e-12)


def test_stance_anchor_recovers_pelvis_translation_and_switches_continuously():
    # Left foot is lower while its pelvis-relative x retreats by one metre.
    feet = np.asarray(
        [
            [[0.0, 0.1, -1.0], [0.0, -0.1, -0.9]],
            [[-0.5, 0.1, -1.0], [0.2, -0.1, -0.9]],
            [[-1.0, 0.1, -0.9], [-0.2, -0.1, -1.0]],
        ]
    )
    position, stance = stance_foot_odometry(feet, np.zeros(3), switch_height_margin_m=0.01)
    assert stance.tolist() == [0, 0, 1]
    np.testing.assert_allclose(position[:2, 0], [0.0, 0.5], atol=1e-12)
    # Switching initializes the new anchor from the previous pelvis position.
    np.testing.assert_allclose(position[2], position[1], atol=1e-12)
