import numpy as np
import pytest

from g1_root_state_bridge.livox_pelvis_yaw import (
    LivoxPelvisYawError,
    quaternion_yaw_z_up,
    recover_pelvis_yaw,
    rehead_planar_trajectory,
    yaw_quaternion_xyzw,
)


def test_recovers_pelvis_heading_after_upside_down_sensor_and_waist_motion():
    time = np.arange(0, 101, dtype=np.int64) * 10_000_000
    seconds = time.astype(np.float64) * 1e-9
    pelvis_rate = 0.4
    waist_rate = 0.2
    bias = -0.03
    # sensor z is opposite torso z: raw = -(pelvis + waist) + bias
    raw_gyro_z = -(pelvis_rate + waist_rate) * np.ones(time.size) + bias
    recovered = recover_pelvis_yaw(
        output_time_ns=time,
        imu_time_ns=time,
        sensor_gyro_z_radps=raw_gyro_z,
        sensor_gyro_z_bias_radps=bias,
        waist_time_ns=time,
        waist_yaw_rad=waist_rate * seconds,
    )
    assert recovered == pytest.approx(pelvis_rate * seconds, abs=1e-12)


def test_fails_closed_on_missing_imu_interval():
    time = np.asarray([0, 10_000_000, 100_000_000], dtype=np.int64)
    with pytest.raises(LivoxPelvisYawError, match="IMU gap"):
        recover_pelvis_yaw(
            output_time_ns=np.asarray([0, 10_000_000], dtype=np.int64),
            imu_time_ns=time,
            sensor_gyro_z_radps=np.zeros(3),
            sensor_gyro_z_bias_radps=0.0,
            waist_time_ns=time,
            waist_yaw_rad=np.zeros(3),
        )


def test_yaw_quaternion_is_z_axis_rotation():
    quaternion = yaw_quaternion_xyzw(np.asarray([0.0, np.pi]))
    assert quaternion[0] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert quaternion[1] == pytest.approx([0.0, 0.0, 1.0, 0.0], abs=1e-12)


def test_reheads_source_local_translation_without_changing_step_lengths():
    position = np.asarray([[0.0, 0.0, 0.8], [1.0, 0.0, 0.8], [2.0, 0.0, 0.8]])
    source_yaw = np.zeros(3)
    replacement = np.full(3, np.pi / 2.0)
    output = rehead_planar_trajectory(position, source_yaw, replacement)
    np.testing.assert_allclose(
        output,
        [[0.0, 0.0, 0.8], [0.0, 1.0, 0.8], [0.0, 2.0, 0.8]],
        atol=1e-12,
    )


def test_quaternion_yaw_unwraps_across_pi():
    yaw = np.asarray([3.0, 3.2, 3.4])
    assert quaternion_yaw_z_up(yaw_quaternion_xyzw(yaw)) == pytest.approx(yaw)
