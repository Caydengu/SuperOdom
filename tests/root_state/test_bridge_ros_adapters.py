from __future__ import annotations

import math

import numpy as np
from geometry_msgs.msg import Transform
from nav_msgs.msg import Odometry

from g1_root_state_bridge.bridge_node import (
    _observation_from_odometry,
    _transform_matrix,
)


def test_typed_odometry_adapter_preserves_exact_superodom_semantics() -> None:
    message = Odometry()
    message.header.stamp.sec = 12
    message.header.stamp.nanosec = 345
    message.header.frame_id = "map"
    message.child_frame_id = "sensor"
    message.pose.pose.position.x = 1.2
    message.pose.pose.position.y = -0.4
    message.pose.pose.position.z = 0.8
    message.pose.pose.orientation.w = math.cos(0.25)
    message.pose.pose.orientation.z = math.sin(0.25)
    message.twist.twist.linear.x = 0.7
    message.twist.twist.linear.y = -0.2
    message.twist.twist.linear.z = 0.1
    message.twist.twist.angular.x = 0.3
    message.twist.twist.angular.y = -0.1
    message.twist.twist.angular.z = 0.4
    message.pose.covariance[0] = 1.0

    observation = _observation_from_odometry(message, receipt_time_ns=12_100_000_000)

    assert observation.estimate_time_ns == 12_000_000_345
    assert observation.receipt_time_ns == 12_100_000_000
    assert observation.frame_id == "map"
    assert observation.child_frame_id == "sensor"
    assert np.allclose(observation.map_T_lidar[:3, 3], (1.2, -0.4, 0.8))
    assert np.allclose(observation.imu_linear_velocity_imu, (0.7, -0.2, 0.1))
    assert np.allclose(observation.imu_angular_velocity_imu, (0.3, -0.1, 0.4))
    assert observation.estimator_healthy


def test_typed_transform_adapter_uses_xyzw_only_at_ros_boundary() -> None:
    message = Transform()
    message.translation.x = -0.011
    message.translation.y = -0.02329
    message.translation.z = 0.04412
    message.rotation.x = 1.0
    message.rotation.w = 0.0

    transform = _transform_matrix(message)

    assert np.allclose(transform[:3, :3], np.diag((1.0, -1.0, -1.0)))
    assert np.allclose(transform[:3, 3], (-0.011, -0.02329, 0.04412))
    assert np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0))
