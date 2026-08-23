import numpy as np

from scripts.run_kiss_icp_scan_archive import (
    apply_rotation_vectors,
    gyro_deskew_scan,
    rotation_to_xyzw,
)


def test_rotation_to_xyzw_round_trip_for_planar_yaw():
    yaw = 1.2
    rotation = np.asarray(
        ((np.cos(yaw), -np.sin(yaw), 0.0), (np.sin(yaw), np.cos(yaw), 0.0), (0.0, 0.0, 1.0))
    )
    quaternion = rotation_to_xyzw(rotation)
    np.testing.assert_allclose(
        quaternion,
        [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)],
        atol=1e-12,
    )


def test_apply_rotation_vectors_rotates_each_point() -> None:
    vectors = np.asarray(((0.0, 0.0, np.pi / 2.0), (0.0, 0.0, 0.0)))
    points = np.asarray(((1.0, 0.0, 0.0), (0.0, 2.0, 0.0)))
    np.testing.assert_allclose(
        apply_rotation_vectors(vectors, points),
        ((0.0, 1.0, 0.0), (0.0, 2.0, 0.0)),
        atol=1e-12,
    )


def test_gyro_deskew_scan_reconstructs_static_points_in_end_frame() -> None:
    # A static world point is observed while the sensor rotates +90 degrees.
    relative_s = np.asarray((0.0, 0.5, 1.0))
    yaw = relative_s * (np.pi / 2.0)
    observed = np.column_stack((np.cos(yaw), -np.sin(yaw), np.zeros(3)))
    imu_time_ns = np.arange(0, 1_000_000_001, 10_000_000, dtype=np.int64)
    imu_gyro = np.tile((0.0, 0.0, np.pi / 2.0), (imu_time_ns.size, 1))
    deskewed, reference_ns, excursion_deg = gyro_deskew_scan(
        observed,
        relative_s,
        scan_start_time_ns=0,
        imu_time_ns=imu_time_ns,
        imu_angular_velocity_radps=imu_gyro,
        gyro_bias_radps=np.zeros(3),
        reference="end",
    )
    np.testing.assert_allclose(deskewed, np.tile((0.0, -1.0, 0.0), (3, 1)), atol=1e-10)
    assert reference_ns == 1_000_000_000
    assert abs(excursion_deg - 90.0) < 1e-9
