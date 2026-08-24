import numpy as np
import pytest
from g1_root_state_bridge.joint_contract import (
    CANONICAL_G1_JOINT_NAMES,
    TimedJointSample,
)
from g1_root_state_bridge.kiss_registration import RegistrationResult
from g1_root_state_bridge.live_pipeline import (
    ImuCoveragePending,
    JointCoveragePending,
    LiveLocalizationConfig,
    LiveLocalizationError,
    SelectedLocalizationPipeline,
)
from g1_root_state_bridge.protocol import (
    REQUIRED_ROOT_FUSION_FLAGS,
    ROOT_STATE_V2_NUM_BYTES,
    RootStateHealth,
)
from g1_root_state_bridge.waist_kinematics import (
    sensor_local_pose_to_pelvis_local_pose,
)


class FakeRegistration:
    def __init__(self) -> None:
        self.index = 0

    def register(self, _points: np.ndarray) -> RegistrationResult:
        pose = np.eye(4)
        pose[0, 3] = 0.1 * self.index
        self.index += 1
        return RegistrationResult(pose, 1.0, 0.2)

    def reset(self) -> None:
        self.index = 0


def _joint(time_ns: int, sequence: int, *, waist_yaw: float = 0.0) -> TimedJointSample:
    position = [0.0] * 29
    position[12] = waist_yaw
    return TimedJointSample(
        stamp_ns=time_ns,
        receipt_ns=time_ns,
        names=CANONICAL_G1_JOINT_NAMES,
        position=tuple(position),
        velocity=(0.0,) * 29,
        sequence=sequence,
    )


def test_pipeline_builds_strict_packet_from_bracketed_sources() -> None:
    pipeline = SelectedLocalizationPipeline(
        FakeRegistration(), LiveLocalizationConfig(gyro_bias_radps=(0.0, 0.0, 0.0))
    )
    start = 1_000_000_000
    for index in range(31):
        stamp = start - 10_000_000 + index * 5_000_000
        pipeline.append_imu(stamp, np.zeros(3))
        pipeline.append_joint(_joint(stamp, index))
    output = pipeline.process_scan(
        np.tile(np.asarray(((1.0, 0.0, 0.0),)), (20, 1)),
        np.linspace(0.0, 0.1, 20),
        scan_start_time_ns=start,
        publish_time_ns=start + 120_000_000,
    )
    assert len(output.payload) == ROOT_STATE_V2_NUM_BYTES
    assert output.packet.health_flags & REQUIRED_ROOT_FUSION_FLAGS == REQUIRED_ROOT_FUSION_FLAGS
    assert output.source_joint_sequence == 22
    assert output.source_joint_time_ns == start + 100_000_000
    np.testing.assert_allclose(output.local_T_pelvis, np.eye(4), atol=1e-12)
    expected_sensor = pipeline._pelvis0_T_sensor0
    expected_cloud = (
        np.tile(np.asarray(((1.0, 0.0, 0.0),)), (20, 1))
        @ expected_sensor[:3, :3].T
        + expected_sensor[:3, 3]
    )
    np.testing.assert_allclose(
        output.registered_points_local_xyz_m,
        expected_cloud,
        atol=1e-12,
    )
    packed = np.frombuffer(output.registered_cloud_xyz32, dtype="<f4").reshape(-1, 3)
    np.testing.assert_allclose(packed, expected_cloud, rtol=1e-6, atol=1e-6)
    assert output.stage_runtime_ms["cloud_transform"] >= 0.0
    assert output.stage_runtime_ms["cloud_pack"] >= 0.0


def test_replay_publish_offset_includes_measured_processing_time() -> None:
    pipeline = SelectedLocalizationPipeline(
        FakeRegistration(), LiveLocalizationConfig(gyro_bias_radps=(0.0, 0.0, 0.0))
    )
    start = 1_500_000_000
    for index in range(31):
        stamp = start - 10_000_000 + index * 5_000_000
        pipeline.append_imu(stamp, np.zeros(3))
        pipeline.append_joint(_joint(stamp, index))
    output = pipeline.process_scan(
        np.tile(np.asarray(((1.0, 0.0, 0.0),)), (20, 1)),
        np.linspace(0.0, 0.1, 20),
        scan_start_time_ns=start,
        publish_time_offset_ns=5_000_000,
    )
    age_ns = output.packet.publish_time_ns - output.packet.estimate_time_ns
    assert age_ns >= 5_000_000
    assert age_ns < 50_000_000


def test_pipeline_keeps_native_translation_and_uses_livox_yaw() -> None:
    pipeline = SelectedLocalizationPipeline(
        FakeRegistration(), LiveLocalizationConfig(gyro_bias_radps=(0.0, 0.0, 0.0))
    )
    start = 2_000_000_000
    for index in range(61):
        stamp = start - 10_000_000 + index * 5_000_000
        gyro = np.asarray((0.0, 0.0, -1.0))
        pipeline.append_imu(stamp, gyro)
        pipeline.append_joint(_joint(stamp, index))
    first = pipeline.process_scan(
        np.tile(np.asarray(((1.0, 0.0, 0.0),)), (20, 1)),
        np.linspace(0.0, 0.1, 20),
        scan_start_time_ns=start,
        publish_time_ns=start + 120_000_000,
    )
    second = pipeline.process_scan(
        np.tile(np.asarray(((1.0, 0.0, 0.0),)), (20, 1)),
        np.linspace(0.0, 0.1, 20),
        scan_start_time_ns=start + 100_000_000,
        publish_time_ns=start + 220_000_000,
    )
    assert first.packet.position[0] == pytest.approx(0.0, abs=1e-12)
    assert second.packet.position[0] > 0.09
    assert second.packet.quaternion_wxyz[3] > 0.0


def test_registered_cloud_and_root_translation_share_initial_pelvis_frame() -> None:
    pipeline = SelectedLocalizationPipeline(
        FakeRegistration(), LiveLocalizationConfig(gyro_bias_radps=(0.0, 0.0, 0.0))
    )
    start = 2_250_000_000
    for index in range(61):
        stamp = start - 10_000_000 + index * 5_000_000
        pipeline.append_imu(stamp, np.zeros(3))
        pipeline.append_joint(_joint(stamp, index))
    points = np.tile(np.asarray(((1.0, 0.0, 0.0),)), (20, 1))
    pipeline.process_scan(
        points,
        np.linspace(0.0, 0.1, 20),
        scan_start_time_ns=start,
        publish_time_ns=start + 120_000_000,
    )
    second = pipeline.process_scan(
        points,
        np.linspace(0.0, 0.1, 20),
        scan_start_time_ns=start + 100_000_000,
        publish_time_ns=start + 220_000_000,
    )
    sensor_pose = np.eye(4)
    sensor_pose[0, 3] = 0.1
    local_T_sensor = pipeline._pelvis0_T_sensor0 @ sensor_pose
    expected_cloud = points @ local_T_sensor[:3, :3].T + local_T_sensor[:3, 3]
    np.testing.assert_allclose(
        second.registered_points_local_xyz_m,
        expected_cloud,
        atol=1e-12,
    )
    expected_pelvis = sensor_local_pose_to_pelvis_local_pose(
        sensor_pose,
        pelvis0_T_sensor0=pipeline._pelvis0_T_sensor0,
        pelvis_t_T_sensor_t=pipeline._pelvis0_T_sensor0,
    )
    np.testing.assert_allclose(second.local_T_pelvis[:3, 3], expected_pelvis[:3, 3])


def test_navigation_heading_does_not_subtract_waist_yaw() -> None:
    pipeline = SelectedLocalizationPipeline(
        FakeRegistration(), LiveLocalizationConfig(gyro_bias_radps=(0.0, 0.0, 0.0))
    )
    start = 2_500_000_000
    for index in range(61):
        stamp = start - 10_000_000 + index * 5_000_000
        pipeline.append_imu(stamp, np.zeros(3))
        pipeline.append_joint(_joint(stamp, index, waist_yaw=0.01 * index))
    first = pipeline.process_scan(
        np.tile(np.asarray(((1.0, 0.0, 0.0),)), (20, 1)),
        np.linspace(0.0, 0.1, 20),
        scan_start_time_ns=start,
        publish_time_ns=start + 120_000_000,
    )
    second = pipeline.process_scan(
        np.tile(np.asarray(((1.0, 0.0, 0.0),)), (20, 1)),
        np.linspace(0.0, 0.1, 20),
        scan_start_time_ns=start + 100_000_000,
        publish_time_ns=start + 220_000_000,
    )
    assert first.packet.quaternion_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert second.packet.quaternion_wxyz == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_unpublished_bootstrap_advances_registration_without_packet() -> None:
    registration = FakeRegistration()
    pipeline = SelectedLocalizationPipeline(
        registration, LiveLocalizationConfig(gyro_bias_radps=(0.0, 0.0, 0.0))
    )
    result = pipeline.bootstrap_unpublished_scan(np.ones((20, 3)))
    assert result.sensor0_T_sensor[0, 3] == 0.0
    assert registration.index == 1


def test_scan_newer_than_latest_imu_is_a_retryable_coverage_error() -> None:
    pipeline = SelectedLocalizationPipeline(
        FakeRegistration(), LiveLocalizationConfig(gyro_bias_radps=(0.0, 0.0, 0.0))
    )
    start = 2_750_000_000
    for index in range(20):
        stamp = start - 10_000_000 + index * 5_000_000
        pipeline.append_imu(stamp, np.zeros(3))
        pipeline.append_joint(_joint(stamp, index))
    with pytest.raises(ImuCoveragePending) as caught:
        pipeline.process_scan(
            np.tile(np.asarray(((1.0, 0.0, 0.0),)), (20, 1)),
            np.linspace(0.0, 0.1, 20),
            scan_start_time_ns=start,
            publish_time_ns=start + 120_000_000,
        )
    assert caught.value.missing_ns == 15_000_000


def test_scan_newer_than_latest_joint_is_retryable_before_registration() -> None:
    registration = FakeRegistration()
    pipeline = SelectedLocalizationPipeline(
        registration, LiveLocalizationConfig(gyro_bias_radps=(0.0, 0.0, 0.0))
    )
    start = 2_875_000_000
    for index in range(25):
        stamp = start - 10_000_000 + index * 5_000_000
        pipeline.append_imu(stamp, np.zeros(3))
        if index < 20:
            pipeline.append_joint(_joint(stamp, index))
    relative = np.linspace(0.0, 0.1, 20)
    with pytest.raises(JointCoveragePending) as caught:
        pipeline.assert_scan_sources_ready(relative, scan_start_time_ns=start)
    assert caught.value.missing_ns == 15_000_000
    assert registration.index == 0


def test_imu_gap_restart_preserves_registration_and_heading_state() -> None:
    registration = FakeRegistration()
    pipeline = SelectedLocalizationPipeline(
        registration, LiveLocalizationConfig(gyro_bias_radps=(0.0, 0.0, 0.0))
    )
    start = 2_900_000_000
    for index in range(10):
        stamp = start + index * 5_000_000
        pipeline.append_imu(stamp, np.asarray((0.0, 0.0, -1.0)))
    pipeline.bootstrap_unpublished_scan(np.ones((20, 3)))
    yaw_before = pipeline.imu._torso_yaw
    pipeline.restart_imu_after_gap(
        start + 100_000_000,
        np.asarray((0.0, 0.0, -1.0)),
    )
    assert pipeline.imu._torso_yaw == yaw_before
    assert registration.index == 1
    assert pipeline._registration_initialized


def test_source_preflight_admits_scan_without_advancing_registration() -> None:
    registration = FakeRegistration()
    pipeline = SelectedLocalizationPipeline(
        registration, LiveLocalizationConfig(gyro_bias_radps=(0.0, 0.0, 0.0))
    )
    start = 2_950_000_000
    for index in range(31):
        stamp = start - 10_000_000 + index * 5_000_000
        pipeline.append_imu(stamp, np.zeros(3))
        pipeline.append_joint(_joint(stamp, index))
    estimate_ns = pipeline.assert_scan_sources_ready(
        np.linspace(0.0, 0.1, 20),
        scan_start_time_ns=start,
    )
    assert estimate_ns == start + 100_000_000
    assert registration.index == 0


def test_reset_discards_registration_and_allows_one_new_bootstrap() -> None:
    registration = FakeRegistration()
    pipeline = SelectedLocalizationPipeline(
        registration, LiveLocalizationConfig(gyro_bias_radps=(0.0, 0.0, 0.0))
    )
    pipeline.bootstrap_unpublished_scan(np.ones((20, 3)))
    with pytest.raises(LiveLocalizationError, match="raw bootstrap"):
        pipeline.bootstrap_unpublished_scan(np.ones((20, 3)))
    pipeline.reset(1)
    result = pipeline.bootstrap_unpublished_scan(np.ones((20, 3)))
    assert result.sensor0_T_sensor[0, 3] == 0.0


def test_untrusted_clock_or_calibration_stays_explicit_in_packet_health() -> None:
    pipeline = SelectedLocalizationPipeline(
        FakeRegistration(), LiveLocalizationConfig(gyro_bias_radps=(0.0, 0.0, 0.0))
    )
    start = 3_000_000_000
    for index in range(31):
        stamp = start - 10_000_000 + index * 5_000_000
        pipeline.append_imu(stamp, np.zeros(3))
        pipeline.append_joint(_joint(stamp, index))
    output = pipeline.process_scan(
        np.tile(np.asarray(((1.0, 0.0, 0.0),)), (20, 1)),
        np.linspace(0.0, 0.1, 20),
        scan_start_time_ns=start,
        publish_time_ns=start + 120_000_000,
        clock_valid=False,
        calibration_valid=False,
    )
    assert not output.packet.health_flags & RootStateHealth.CLOCK_VALID
    assert not output.packet.health_flags & RootStateHealth.CALIBRATION_VALID
    assert not output.packet.strictly_valid
