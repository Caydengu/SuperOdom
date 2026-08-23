import numpy as np
import pytest

from g1_root_state_bridge.livox_deskew import LivoxDeskewError, gyro_deskew_scan


def test_zero_rate_deskew_preserves_points_and_uses_scan_end() -> None:
    points = np.asarray(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    relative = np.asarray((0.0, 0.1))
    imu_time = np.asarray((1_000_000_000, 1_050_000_000, 1_100_000_000), dtype=np.int64)
    deskewed, stamp, excursion = gyro_deskew_scan(
        points,
        relative,
        scan_start_time_ns=1_000_000_000,
        imu_time_ns=imu_time,
        imu_angular_velocity_radps=np.zeros((3, 3)),
        gyro_bias_radps=np.zeros(3),
    )
    np.testing.assert_allclose(deskewed, points)
    assert stamp == 1_100_000_000
    assert excursion == pytest.approx(0.0)


def test_missing_imu_coverage_fails_closed() -> None:
    with pytest.raises(LivoxDeskewError, match="ends too early"):
        gyro_deskew_scan(
            np.asarray(((1.0, 0.0, 0.0),)),
            np.asarray((0.1,)),
            scan_start_time_ns=1_000_000_000,
            imu_time_ns=np.asarray((1_000_000_000, 1_010_000_000), dtype=np.int64),
            imu_angular_velocity_radps=np.zeros((2, 3)),
            gyro_bias_radps=np.zeros(3),
        )
