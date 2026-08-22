from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from g1_root_state_bridge.bridge_core import (
    BridgeCore,
    BridgeCoreError,
    CorrectionSample,
    EstimatorCalibration,
    EstimatorHealthSample,
    EstimatorObservation,
    SuperOdometryObservationNormalizer,
)
from g1_root_state_bridge.joint_contract import (
    CANONICAL_G1_JOINT_NAMES,
    TimedJointSample,
)
from g1_root_state_bridge.kinematics import PelvisState
from g1_root_state_bridge.odometry_health import OdometryHealthConfig
from g1_root_state_bridge.orientation_fusion import OrientationFusionConfig
from g1_root_state_bridge.protocol import REQUIRED_HEALTH_FLAGS, RootStateHealth
from g1_root_state_bridge.root_imu_contract import TimedRootImuSample


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
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _calibration(
    *,
    map_T_gravity: np.ndarray | None = None,
    imu_T_lidar: np.ndarray | None = None,
    digest: bytes = b"c" * 32,
    valid: bool = True,
) -> EstimatorCalibration:
    return EstimatorCalibration(
        semantics_version="superodom-state-estimation-raw-v1",
        map_frame="map",
        sensor_frame="sensor",
        gravity_frame="gravity_aligned",
        initialization_time_ns=900_000_000,
        map_T_gravity=np.eye(4) if map_T_gravity is None else map_T_gravity,
        imu_T_lidar=np.eye(4) if imu_T_lidar is None else imu_T_lidar,
        calibration_digest=digest,
        valid=valid,
    )


def _observation(**mutations: object) -> EstimatorObservation:
    base = EstimatorObservation(
        estimate_time_ns=1_000_000_000,
        receipt_time_ns=1_001_000_000,
        frame_id="map",
        child_frame_id="sensor",
        map_T_lidar=_transform(
            _rot_z(0.32) @ _rot_y(-0.11),
            (0.8, -0.4, 1.1),
        ),
        imu_linear_velocity_imu=np.array((0.7, -0.2, 0.1)),
        imu_angular_velocity_imu=np.array((0.3, -0.1, 0.4)),
        estimator_healthy=True,
    )
    return replace(base, **mutations)


def _joint(stamp_ns: int, sequence: int, waist_offset: float = 0.0) -> TimedJointSample:
    q = [0.0] * 29
    dq = [0.0] * 29
    for index in (12, 13, 14):
        q[index] = waist_offset + 0.01 * index
        dq[index] = -0.1 * waist_offset + 0.001 * index
    return TimedJointSample(
        stamp_ns=stamp_ns,
        receipt_ns=stamp_ns + 1_000_000,
        names=CANONICAL_G1_JOINT_NAMES,
        position=tuple(q),
        velocity=tuple(dq),
        sequence=sequence,
        source_epoch=5,
    )


class IdentityKinematics:
    def compute(
        self,
        *,
        world_T_observed: np.ndarray,
        observed_linear_velocity_world: np.ndarray,
        observed_angular_velocity_world: np.ndarray,
        joint_sample: TimedJointSample,
        physical_sensor_T_observed: np.ndarray,
    ) -> PelvisState:
        assert joint_sample.names == CANONICAL_G1_JOINT_NAMES
        assert np.allclose(physical_sensor_T_observed, np.eye(4))
        return PelvisState(
            world_T_pelvis=world_T_observed,
            linear_velocity_world=observed_linear_velocity_world,
            angular_velocity_world=observed_angular_velocity_world,
        )


def _ready_core(
    *,
    calibration: EstimatorCalibration | None = None,
    allowed_digests: set[bytes] | None = None,
    max_correction_input_age_ns: int = 150_000_000,
    max_correction_age_ns: int = 250_000_000,
    odometry_health_config: OdometryHealthConfig | None = None,
    orientation_fusion_config: OrientationFusionConfig | None = None,
) -> BridgeCore:
    selected = _calibration() if calibration is None else calibration
    core = BridgeCore(
        calibration=selected,
        kinematics=IdentityKinematics(),
        source_epoch=17,
        allowed_calibration_digests=(
            {selected.calibration_digest} if allowed_digests is None else allowed_digests
        ),
        max_joint_gap_ns=10_000_000,
        max_correction_input_age_ns=max_correction_input_age_ns,
        max_correction_age_ns=max_correction_age_ns,
        max_health_age_ns=20_000_000,
        odometry_health_config=odometry_health_config,
        orientation_fusion_config=orientation_fusion_config,
    )
    core.append_joint(_joint(995_000_000, 10, -0.1))
    core.append_joint(_joint(1_006_000_000, 11, 0.1))
    core.update_correction(
        CorrectionSample(
            reference_time_ns=850_000_000,
            evidence_time_ns=900_000_000,
            application_time_ns=900_500_000,
            receipt_time_ns=901_000_000,
            sequence=3,
            reset_id=1,
            covariance_diagonal=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
        )
    )
    core.update_health(EstimatorHealthSample(receipt_time_ns=1_002_000_000, healthy=True))
    return core


def _root_imu(stamp_ns: int, sequence: int) -> TimedRootImuSample:
    return TimedRootImuSample(
        stamp_ns=stamp_ns,
        receipt_ns=stamp_ns + 1_000_000,
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.4),
        linear_acceleration=(0.0, 0.0, 9.81),
        sequence=sequence,
        source_epoch=5,
    )


def test_optional_root_imu_fusion_is_source_synchronized_and_explicitly_healthy() -> None:
    core = _ready_core(orientation_fusion_config=OrientationFusionConfig())
    core.append_root_imu(_root_imu(995_000_000, 1))
    core.append_root_imu(_root_imu(1_005_000_000, 2))
    packet = core.build_packet(_observation(), publish_time_ns=1_003_000_000)

    assert packet.health_flags & RootStateHealth.ROOT_IMU_SYNC_VALID
    assert packet.health_flags & RootStateHealth.ROOT_ORIENTATION_FUSED
    assert packet.health_flags & RootStateHealth.ESTIMATOR_HEALTHY


def test_required_root_imu_fusion_drops_packet_when_no_imu_bracket_exists() -> None:
    core = _ready_core(orientation_fusion_config=OrientationFusionConfig())
    with pytest.raises(BridgeCoreError, match="root IMU synchronization failed"):
        core.build_packet(_observation(), publish_time_ns=1_003_000_000)


def test_optional_odometry_gate_clears_estimator_health_until_plausible() -> None:
    core = _ready_core(
        odometry_health_config=OdometryHealthConfig(
            window_ns=100_000_000,
            max_planar_speed_mps=1.5,
            recovery_hold_ns=0,
        )
    )

    warming = core.build_packet(
        _observation(), publish_time_ns=1_003_000_000
    )
    core.append_joint(_joint(1_095_000_000, 12, -0.1))
    core.append_joint(_joint(1_106_000_000, 13, 0.1))
    core.update_health(EstimatorHealthSample(receipt_time_ns=1_102_000_000, healthy=True))
    implausible = core.build_packet(
        _observation(
            estimate_time_ns=1_100_000_000,
            receipt_time_ns=1_101_000_000,
            map_T_lidar=_transform(np.eye(3), (0.3, -0.4, 1.1)),
        ),
        publish_time_ns=1_103_000_000,
    )

    assert not (warming.health_flags & RootStateHealth.ESTIMATOR_HEALTHY)
    assert core.last_odometry_health is not None
    assert "planar_speed" in core.last_odometry_health.reason
    assert not (implausible.health_flags & RootStateHealth.ESTIMATOR_HEALTHY)


def test_normalizer_converts_raw_map_and_imu_origin_twist_to_gravity_lidar() -> None:
    map_T_gravity = _transform(_rot_x(math.pi), (0.0, 0.0, 0.0))
    imu_T_lidar = _transform(np.eye(3), (-0.011, -0.02329, 0.04412))
    calibration = _calibration(
        map_T_gravity=map_T_gravity,
        imu_T_lidar=imu_T_lidar,
    )
    observation = _observation()

    normalized = SuperOdometryObservationNormalizer(calibration).normalize(observation)

    gravity_T_map = np.linalg.inv(map_T_gravity)
    expected_pose = gravity_T_map @ observation.map_T_lidar
    map_T_imu = observation.map_T_lidar @ np.linalg.inv(imu_T_lidar)
    map_R_imu = map_T_imu[:3, :3]
    omega_map = map_R_imu @ observation.imu_angular_velocity_imu
    velocity_imu_map = map_R_imu @ observation.imu_linear_velocity_imu
    imu_to_lidar_map = map_R_imu @ imu_T_lidar[:3, 3]
    velocity_lidar_map = velocity_imu_map + np.cross(omega_map, imu_to_lidar_map)
    expected_linear = gravity_T_map[:3, :3] @ velocity_lidar_map
    expected_angular = gravity_T_map[:3, :3] @ omega_map

    assert np.allclose(normalized.world_T_lidar, expected_pose, atol=1e-12)
    assert np.allclose(normalized.lidar_linear_velocity_world, expected_linear, atol=1e-12)
    assert np.allclose(normalized.lidar_angular_velocity_world, expected_angular, atol=1e-12)
    assert normalized.estimate_time_ns == observation.estimate_time_ns


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"frame_id": "odom"}, "frame"),
        ({"child_frame_id": "base_link"}, "sensor"),
        ({"map_T_lidar": np.zeros((4, 4))}, "transform"),
    ],
)
def test_normalizer_rejects_frame_or_transform_contract_mismatch(
    mutation: dict[str, object], match: str
) -> None:
    normalizer = SuperOdometryObservationNormalizer(_calibration())
    with pytest.raises(BridgeCoreError, match=match):
        normalizer.normalize(_observation(**mutation))


def test_bridge_emits_strict_packet_with_real_joint_gap_and_correction_metadata() -> None:
    core = _ready_core()

    packet = core.build_packet(_observation(), publish_time_ns=1_007_000_000)

    assert packet.sequence == 1
    assert packet.source_epoch == 17
    assert packet.estimate_time_ns == 1_000_000_000
    assert packet.publish_time_ns == 1_007_000_000
    assert packet.correction_time_ns == 900_000_000
    assert packet.joint_time_ns == 995_000_000
    assert packet.joint_sync_gap_ns == -5_000_000
    assert packet.health_flags == REQUIRED_HEALTH_FLAGS
    assert packet.strictly_valid
    assert packet.covariance_diagonal == (0.01, 0.02, 0.03, 0.04, 0.05, 0.06)
    assert packet.calibration_digest == b"c" * 32


def test_bridge_exposes_the_exact_synchronized_joint_evidence_used_for_fk() -> None:
    core = _ready_core()

    result = core.build_packet_with_evidence(
        _observation(),
        publish_time_ns=1_007_000_000,
    )

    assert result.packet.strictly_valid
    assert result.joint_sample.stamp_ns == result.packet.estimate_time_ns
    assert result.joint_sample.names == CANONICAL_G1_JOINT_NAMES
    assert result.joint_sample.position[12:15] == pytest.approx(
        (0.1109090909, 0.1209090909, 0.1309090909)
    )


def test_bridge_estimator_health_fails_independently_without_reusing_good_state() -> None:
    core = _ready_core()
    core.append_joint(_joint(1_095_000_000, 12, -0.1))
    core.append_joint(_joint(1_106_000_000, 13, 0.1))
    core.update_health(EstimatorHealthSample(receipt_time_ns=1_103_000_000, healthy=False))

    packet = core.build_packet(
        _observation(estimate_time_ns=1_100_000_000, receipt_time_ns=1_101_000_000),
        publish_time_ns=1_107_000_000,
    )

    assert not (packet.health_flags & RootStateHealth.ESTIMATOR_HEALTHY)
    assert packet.health_flags & RootStateHealth.CORRECTION_FRESH
    assert packet.health_flags & RootStateHealth.FINITE_POSE
    assert packet.health_flags & RootStateHealth.JOINT_SYNC_VALID
    assert packet.health_flags & RootStateHealth.CALIBRATION_VALID
    assert packet.health_flags & RootStateHealth.CLOCK_VALID
    assert not packet.strictly_valid


def test_correction_quality_and_hold_horizons_are_independent() -> None:
    core = _ready_core()

    held_packet = core.build_packet(
        _observation(),
        publish_time_ns=1_120_000_000,
    )
    expired_packet = core.build_packet(
        _observation(),
        publish_time_ns=1_150_000_001,
    )

    assert held_packet.health_flags & RootStateHealth.CORRECTION_FRESH
    assert not (expired_packet.health_flags & RootStateHealth.CORRECTION_FRESH)


def test_late_arriving_correction_is_not_fresh_even_inside_hold_horizon() -> None:
    core = _ready_core()
    core._corrections.clear()
    core.update_correction(
        CorrectionSample(
            reference_time_ns=800_000_000,
            evidence_time_ns=900_000_000,
            application_time_ns=1_050_000_000,
            receipt_time_ns=1_051_000_001,
            sequence=8,
            reset_id=1,
            covariance_diagonal=(0.1,) * 6,
        )
    )

    packet = core.build_packet(
        _observation(receipt_time_ns=1_060_000_000),
        publish_time_ns=1_070_000_000,
    )

    assert not (packet.health_flags & RootStateHealth.CORRECTION_FRESH)


def test_bridge_marks_unallowlisted_or_invalid_calibration_unhealthy() -> None:
    calibration = _calibration(valid=False)
    core = _ready_core(calibration=calibration, allowed_digests={b"x" * 32})

    packet = core.build_packet(_observation(), publish_time_ns=1_007_000_000)

    assert not (packet.health_flags & RootStateHealth.CALIBRATION_VALID)
    assert not packet.strictly_valid


def test_bridge_reset_changes_epoch_and_restarts_sequence() -> None:
    core = _ready_core()
    first = core.build_packet(_observation(), publish_time_ns=1_007_000_000)
    second = core.build_packet(_observation(), publish_time_ns=1_008_000_000)
    core.reset(source_epoch=18)
    core.append_joint(_joint(995_000_000, 1, -0.1))
    core.append_joint(_joint(1_006_000_000, 2, 0.1))
    core.update_correction(
        CorrectionSample(
            reference_time_ns=850_000_000,
            evidence_time_ns=900_000_000,
            application_time_ns=900_500_000,
            receipt_time_ns=901_000_000,
            sequence=1,
            reset_id=1,
            covariance_diagonal=(0.1,) * 6,
        )
    )
    core.update_health(EstimatorHealthSample(receipt_time_ns=1_002_000_000, healthy=True))
    restarted = core.build_packet(_observation(), publish_time_ns=1_007_000_000)

    assert (first.sequence, second.sequence) == (1, 2)
    assert restarted.source_epoch == 18
    assert restarted.sequence == 1


def test_bridge_uses_newest_lidar_evidence_not_scan_start_for_freshness() -> None:
    core = _ready_core()
    core._corrections.clear()
    core.update_correction(
        CorrectionSample(
            reference_time_ns=800_000_000,
            evidence_time_ns=900_000_000,
            application_time_ns=900_500_000,
            receipt_time_ns=901_000_000,
            sequence=8,
            reset_id=1,
            covariance_diagonal=(0.1,) * 6,
        )
    )

    packet = core.build_packet(_observation(), publish_time_ns=1_007_000_000)

    assert packet.correction_time_ns == 900_000_000
    assert packet.health_flags & RootStateHealth.CORRECTION_FRESH


def test_bridge_does_not_attribute_a_correction_to_preapplication_state() -> None:
    core = _ready_core()
    core._corrections.clear()
    core.update_correction(
        CorrectionSample(
            reference_time_ns=800_000_000,
            evidence_time_ns=900_000_000,
            application_time_ns=1_002_000_000,
            receipt_time_ns=1_003_000_000,
            sequence=8,
            reset_id=1,
            covariance_diagonal=(0.1,) * 6,
        )
    )

    packet = core.build_packet(_observation(), publish_time_ns=1_007_000_000)

    assert packet.correction_time_ns == 0
    assert not (packet.health_flags & RootStateHealth.CORRECTION_FRESH)


def test_bridge_refuses_missing_joint_bracket_instead_of_reusing_previous_root() -> None:
    core = _ready_core()
    core.build_packet(_observation(), publish_time_ns=1_007_000_000)

    with pytest.raises(BridgeCoreError, match="joint"):
        core.build_packet(
            _observation(estimate_time_ns=1_100_000_000, receipt_time_ns=1_101_000_000),
            publish_time_ns=1_102_000_000,
        )
