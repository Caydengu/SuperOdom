from __future__ import annotations

import numpy as np
import pytest

from g1_root_state_bridge.replay import gravity_alignment_from_accelerations


def test_gravity_alignment_recovers_z_up_for_upside_down_sensor() -> None:
    map_T_lidar = np.eye(4)
    imu_T_lidar = np.eye(4)
    accelerations = [np.array((0.0, 0.0, -9.81)) for _ in range(100)]

    map_T_gravity = gravity_alignment_from_accelerations(
        accelerations,
        map_T_lidar=map_T_lidar,
        imu_T_lidar=imu_T_lidar,
    )

    assert np.allclose(map_T_gravity[:3, :3], np.diag((1.0, -1.0, -1.0)))
    assert np.allclose(map_T_gravity[:3, 3], 0.0)
    assert np.linalg.det(map_T_gravity[:3, :3]) == pytest.approx(1.0)


def test_gravity_alignment_rejects_unobservable_acceleration() -> None:
    with pytest.raises(ValueError, match="gravity"):
        gravity_alignment_from_accelerations(
            [np.zeros(3)],
            map_T_lidar=np.eye(4),
            imu_T_lidar=np.eye(4),
        )
