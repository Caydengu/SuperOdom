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


def test_lidar_pipeline_events_instrument_the_existing_sole_subscriber() -> None:
    message = _read("super_odometry_msgs/msg/LidarPipelineEvent.msg")
    for field in (
        "uint8 RAW_RECEIVED=1",
        "uint8 RAW_SKIPPED=2",
        "uint8 FEATURE_PUBLISHED=3",
        "uint8 MAPPING_RECEIVED=4",
        "uint8 MAPPING_QUEUE_DROPPED=5",
        "uint8 CORRECTION_PUBLISHED=6",
        "uint8 MAPPING_REJECTED=7",
        "std_msgs/Header header",
        "string semantics_version",
        "uint8 event_type",
        "uint64 input_count",
        "uint64 output_count",
        "uint64 dropped_count",
        "uint32 queue_depth",
        "builtin_interfaces/Time scan_reference_stamp",
        "builtin_interfaces/Time newest_observation_stamp",
    ):
        assert field in message

    message_cmake = _read("super_odometry_msgs/CMakeLists.txt")
    assert '"msg/LidarPipelineEvent.msg"' in message_cmake

    feature_header = _read(
        "super_odometry/include/super_odometry/FeatureExtraction/featureExtraction.h"
    )
    feature_source = _read(
        "super_odometry/src/FeatureExtraction/featureExtraction.cpp"
    )
    mapping_header = _read(
        "super_odometry/include/super_odometry/LaserMapping/laserMapping.h"
    )
    mapping_source = _read("super_odometry/src/LaserMapping/laserMapping.cpp")
    shadow = _read("docker/humble-minimal/root_state_shadow.sh")

    assert "pubLidarPipelineEvent" in feature_header
    assert "RAW_RECEIVED" in feature_source
    assert "FEATURE_PUBLISHED" in feature_source
    assert "pubLidarPipelineEvent" in mapping_header
    assert "MAPPING_RECEIVED" in mapping_source
    assert "MAPPING_REJECTED" in mapping_source
    assert "CORRECTION_PUBLISHED" in mapping_source
    assert "mappingDroppedCount" in mapping_source
    assert 'ProjectName+"/lidar_pipeline_events"' in feature_source
    assert 'ProjectName+"/lidar_pipeline_events"' in mapping_source
    assert "/lidar_pipeline_events" in shadow
    assert (
        feature_source.count(
            "create_subscription<livox_ros_driver2::msg::CustomMsg>"
        )
        == 1
    )


def test_mapping_drains_feature_queue_without_clearing_newer_frames() -> None:
    mapping_source = _read("super_odometry/src/LaserMapping/laserMapping.cpp")
    process = mapping_source.split("void laserMapping::process()", maxsplit=1)[1]

    assert "sensorMeas=extractSensorData();" in process
    assert "clearSensorData();" not in process
    assert "std::lock_guard<std::mutex> lock(mBuf)" in process
    assert "realsenseBuf.pop();" in mapping_source


def test_feature_to_mapping_transport_buffers_transient_callback_jitter() -> None:
    feature_source = _read(
        "super_odometry/src/FeatureExtraction/featureExtraction.cpp"
    )
    mapping_source = _read("super_odometry/src/LaserMapping/laserMapping.cpp")
    qos_contract = "rclcpp::QoS(rclcpp::KeepLast(16)).reliable()"

    assert qos_contract in feature_source
    assert qos_contract in mapping_source
    assert (
        'create_publisher<super_odometry_msgs::msg::LaserFeature>('
        in feature_source
    )
    assert (
        'create_subscription<super_odometry_msgs::msg::LaserFeature>('
        in mapping_source
    )


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
    assert "serialize_policy_state_v1" in node
    assert '"policy_state_bind_endpoint", "tcp://*:5576"' in node
    assert "self._policy_state_socket" in node
    assert "policy_state_payload_hex" in node
    assert "policy_state_packet_published" in node
    assert "def _send_nonblocking(" in node
    assert "_latest_health_receipt_ns < observation.receipt_time_ns" in node


def test_bridge_default_path_is_recorder_free_with_compact_heartbeat() -> None:
    node = _read("g1_root_state_bridge/g1_root_state_bridge/bridge_node.py")
    assert 'self.declare_parameter("replay_jsonl_path", "")' in node
    assert "if replay_path:" in node
    assert "self._replay_file = None" in node
    assert "self._zmq_socket.setsockopt(zmq.SNDHWM, 1)" in node
    assert "self._policy_state_socket.setsockopt(zmq.SNDHWM, 1)" in node
    assert "self.create_timer(1.0, self._publish_heartbeat)" in node
    assert "self._emit_packet_diagnostics = self._replay_file is not None" in node

    heartbeat = node.split(
        "def _publish_heartbeat",
        maxsplit=1,
    )[1].split("def _on_calibration", maxsplit=1)[0]
    for counter in (
        "packets_built",
        "root_packets_sent",
        "policy_packets_sent",
        "root_packets_dropped",
        "policy_packets_dropped",
    ):
        assert counter in heartbeat
    for forbidden in (
        "payload_hex",
        "joint_position",
        "joint_velocity",
        "quaternion_wxyz",
    ):
        assert forbidden not in heartbeat


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
