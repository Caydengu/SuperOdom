from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def _read(relative: str) -> str:
    return (REPO / relative).read_text(encoding="utf-8")


def test_superodometry_exposes_one_latched_typed_alignment_contract() -> None:
    message = _read("super_odometry_msgs/msg/StateEstimationCalibration.msg")
    for field in (
        "std_msgs/Header header",
        "string semantics_version",
        "string map_frame",
        "string sensor_frame",
        "string gravity_frame",
        "geometry_msgs/Transform map_t_gravity",
        "geometry_msgs/Transform imu_t_lidar",
        "uint64 reset_id",
        "bool valid",
    ):
        assert field in message

    message_cmake = _read("super_odometry_msgs/CMakeLists.txt")
    assert '"msg/StateEstimationCalibration.msg"' in message_cmake

    header = _read(
        "super_odometry/include/super_odometry/ImuPreintegration/imuPreintegration_current.h"
    )
    source = _read(
        "super_odometry/src/ImuPreintegration/imuPreintegration_current.cpp"
    )
    assert "StateEstimationCalibration" in header
    assert "pubStateEstimationCalibration" in header
    assert "transient_local" in source
    assert 'ProjectName+"/state_estimation_calibration"' in source
    assert "map_R_gravity" in source
    assert "g_world_dir" in source
    assert "publishStateEstimationCalibration" in source


def test_superodometry_exposes_explicit_scan_and_applied_correction_times() -> None:
    laser_feature = _read("super_odometry_msgs/msg/LaserFeature.msg")
    assert "builtin_interfaces/Time newest_observation_stamp" in laser_feature

    lidar_correction = _read("super_odometry_msgs/msg/LidarCorrection.msg")
    for field in (
        "string semantics_version",
        "uint64 sequence",
        "builtin_interfaces/Time newest_observation_stamp",
        "builtin_interfaces/Time output_stamp",
        "nav_msgs/Odometry odometry",
    ):
        assert field in lidar_correction

    state_correction = _read(
        "super_odometry_msgs/msg/StateEstimationCorrection.msg"
    )
    for field in (
        "std_msgs/Header header",
        "string semantics_version",
        "uint64 sequence",
        "builtin_interfaces/Time newest_observation_stamp",
        "builtin_interfaces/Time mapping_output_stamp",
        "builtin_interfaces/Time application_stamp",
        "uint64 reset_id",
        "bool valid",
    ):
        assert field in state_correction

    message_cmake = _read("super_odometry_msgs/CMakeLists.txt")
    assert '"msg/LidarCorrection.msg"' in message_cmake
    assert '"msg/StateEstimationCorrection.msg"' in message_cmake

    feature_source = _read(
        "super_odometry/src/FeatureExtraction/featureExtraction.cpp"
    )
    mapping_source = _read("super_odometry/src/LaserMapping/laserMapping.cpp")
    preintegration_source = _read(
        "super_odometry/src/ImuPreintegration/imuPreintegration_current.cpp"
    )
    assert "newest_observation_stamp" in feature_source
    assert "pubLidarCorrection" in mapping_source
    assert 'ProjectName+"/lidar_correction"' in mapping_source
    assert "publishStateEstimationCorrection" in preintegration_source
    assert 'ProjectName+"/state_estimation_correction"' in preintegration_source


def test_bridge_is_an_ament_python_package_with_typed_ros_inputs() -> None:
    package = _read("g1_root_state_bridge/package.xml")
    setup = _read("g1_root_state_bridge/setup.py")
    node = _read("g1_root_state_bridge/g1_root_state_bridge/bridge_node.py")

    for dependency in (
        "rclpy",
        "nav_msgs",
        "std_msgs",
        "super_odometry_msgs",
    ):
        assert f"<exec_depend>{dependency}</exec_depend>" in package
    assert "g1-root-state-bridge" in setup
    assert "nav_msgs.msg import Odometry" in node
    assert "std_msgs.msg import Bool" in node
    assert "OptimizationStats" in node
    assert "StateEstimationCorrection" in node
    assert "StateEstimationCalibration" in node
    assert '"/state_estimation"' in node
    assert '"/super_odometry_stats"' in node
    assert '"/state_estimation_correction"' in node
    assert '"/state_estimation_health"' in node
    assert '"/state_estimation_calibration"' in node
    assert "serialize_root_state_v2" in node
    assert "_latest_health_receipt_ns < observation.receipt_time_ns" in node


def test_bridge_source_has_no_raw_cloud_or_latest_tf_path() -> None:
    node = _read("g1_root_state_bridge/g1_root_state_bridge/bridge_node.py").lower()
    for forbidden in (
        "/livox/lidar",
        "pointcloud2",
        "lookup_transform",
        "time()",
        "frame_normalizer",
        "deserialize_cdr",
    ):
        assert forbidden not in node
