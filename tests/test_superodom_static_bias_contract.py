from pathlib import Path


def test_initial_system_preserves_stationary_gyro_bias() -> None:
    source = Path(
        "super_odometry/src/ImuPreintegration/imuPreintegration_current.cpp"
    ).read_text(encoding="utf-8")
    section = source[source.index("void imuPreintegration::initial_system") :]
    section = section[
        : section.index("void imuPreintegration::publishStateEstimationCalibration")
    ]
    assert "imu_Init->gyr_bias.x()" in section
    assert "prevBias_ = gtsam::imuBias::ConstantBias();" not in section
    assert "resetIntegrationAndSetBias(prevBias_)" in section
