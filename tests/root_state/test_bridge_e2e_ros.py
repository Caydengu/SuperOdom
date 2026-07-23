from __future__ import annotations

import json
import socket
import time
from pathlib import Path

import pytest


rclpy = pytest.importorskip("rclpy")
zmq = pytest.importorskip("zmq")

from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool
from super_odometry_msgs.msg import (
    StateEstimationCalibration,
    StateEstimationCorrection,
)

from g1_root_state_bridge.bridge_node import G1RootStateBridgeNode
from g1_root_state_bridge.joint_contract import CANONICAL_G1_JOINT_NAMES
from g1_root_state_bridge.joint_transport import (
    JointHealth,
    JointPacketV1,
    canonical_joint_mapping_digest,
    serialize_joint_packet,
)
from g1_root_state_bridge.protocol import deserialize_root_state_v2


def _free_port(socket_type: int) -> int:
    sock = socket.socket(socket.AF_INET, socket_type)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _stamp(message_stamp: object, stamp_ns: int) -> None:
    message_stamp.sec = stamp_ns // 1_000_000_000
    message_stamp.nanosec = stamp_ns % 1_000_000_000


def _spin_until(executor: SingleThreadedExecutor, predicate, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for synthetic bridge transaction")
        executor.spin_once(timeout_sec=0.005)


def test_typed_ros_and_udp_inputs_produce_one_strict_zmq_v2_packet(
    tmp_path: Path,
) -> None:
    udp_port = _free_port(socket.SOCK_DGRAM)
    tcp_port = _free_port(socket.SOCK_STREAM)
    endpoint = f"tcp://127.0.0.1:{tcp_port}"
    replay_path = tmp_path / "live_packets.jsonl"

    rclpy.init()
    bridge = G1RootStateBridgeNode(
        parameter_overrides=[
            Parameter("joint_bind_host", value="127.0.0.1"),
            Parameter("joint_bind_port", value=udp_port),
            Parameter("root_state_bind_endpoint", value=endpoint),
            Parameter("max_joint_transport_age_ms", value=100.0),
            Parameter("max_joint_sync_gap_ms", value=20.0),
            Parameter("pending_state_timeout_ms", value=100.0),
            Parameter("replay_jsonl_path", value=str(replay_path)),
        ]
    )
    harness = rclpy.create_node("synthetic_root_state_harness")
    executor = SingleThreadedExecutor()
    executor.add_node(bridge)
    executor.add_node(harness)

    reliable = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=64,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    latched = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )
    calibration_pub = harness.create_publisher(
        StateEstimationCalibration,
        "/state_estimation_calibration",
        latched,
    )
    correction_pub = harness.create_publisher(
        StateEstimationCorrection,
        "/state_estimation_correction",
        reliable,
    )
    health_pub = harness.create_publisher(
        Bool,
        "/state_estimation_health",
        reliable,
    )
    odometry_pub = harness.create_publisher(
        Odometry,
        "/state_estimation",
        reliable,
    )

    zmq_context = zmq.Context()
    subscriber = zmq_context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.LINGER, 0)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.connect(endpoint)
    joint_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        # Allow ROS discovery and the ZMQ PUB/SUB subscription handshake to settle.
        _spin_until(
            executor,
            lambda: calibration_pub.get_subscription_count() == 1
            and odometry_pub.get_subscription_count() == 1,
            2.0,
        )
        time.sleep(0.1)

        base_ns = time.time_ns()
        calibration = StateEstimationCalibration()
        _stamp(calibration.header.stamp, base_ns - 100_000_000)
        calibration.header.frame_id = "map"
        calibration.semantics_version = "superodom-state-estimation-raw-v1"
        calibration.map_frame = "map"
        calibration.sensor_frame = "sensor"
        calibration.gravity_frame = "gravity_aligned"
        calibration.map_t_gravity.rotation.w = 1.0
        calibration.imu_t_lidar.translation.x = float(
            bridge._calibration_contract.imu_T_lidar[0, 3]
        )
        calibration.imu_t_lidar.translation.y = float(
            bridge._calibration_contract.imu_T_lidar[1, 3]
        )
        calibration.imu_t_lidar.translation.z = float(
            bridge._calibration_contract.imu_T_lidar[2, 3]
        )
        calibration.imu_t_lidar.rotation.w = 1.0
        calibration.reset_id = 1
        calibration.valid = True
        calibration_pub.publish(calibration)
        _spin_until(executor, lambda: bridge._core is not None, 2.0)

        estimate_ns = time.time_ns() - 4_000_000
        all_joint_health = (
            JointHealth.SOURCE_TYPED
            | JointHealth.FINITE
            | JointHealth.MOTOR_COUNT_VALID
            | JointHealth.CLOCK_VALID
        )
        for sequence, offset_ns in ((1, -2_000_000), (2, 2_000_000)):
            packet = JointPacketV1(
                source_epoch=17,
                sequence=sequence,
                stamp_ns=estimate_ns + offset_ns,
                source_tick=sequence,
                health_flags=all_joint_health,
                position=(0.0,) * len(CANONICAL_G1_JOINT_NAMES),
                velocity=(0.0,) * len(CANONICAL_G1_JOINT_NAMES),
                mapping_digest=canonical_joint_mapping_digest(),
            )
            joint_sender.sendto(
                serialize_joint_packet(packet),
                ("127.0.0.1", udp_port),
            )
        _spin_until(
            executor,
            lambda: len(bridge._core._joint_buffer._samples) == 2,
            1.0,
        )

        correction = StateEstimationCorrection()
        _stamp(correction.header.stamp, estimate_ns - 101_000_000)
        correction.header.frame_id = "map"
        correction.semantics_version = "superodom-state-correction-v1"
        correction.sequence = 7
        _stamp(correction.newest_observation_stamp, estimate_ns - 1_000_000)
        _stamp(correction.mapping_output_stamp, estimate_ns - 500_000)
        _stamp(correction.application_stamp, estimate_ns - 250_000)
        correction.reset_id = 1
        correction.valid = True
        correction_pub.publish(correction)
        _spin_until(executor, lambda: len(bridge._core._corrections) == 1, 1.0)

        odometry = Odometry()
        _stamp(odometry.header.stamp, estimate_ns)
        odometry.header.frame_id = "map"
        odometry.child_frame_id = "sensor"
        odometry.pose.pose.orientation.w = 1.0
        odometry.pose.covariance[0] = 1.0
        odometry_pub.publish(odometry)
        _spin_until(executor, lambda: len(bridge._pending) == 1, 1.0)

        health = Bool()
        health.data = True
        health_pub.publish(health)

        poller = zmq.Poller()
        poller.register(subscriber, zmq.POLLIN)
        payloads: list[bytes] = []

        def received_packet() -> bool:
            executor.spin_once(timeout_sec=0.005)
            if subscriber in dict(poller.poll(timeout=0)):
                payloads.append(subscriber.recv())
            return bool(payloads)

        _spin_until(executor, received_packet, 2.0)
        packet = deserialize_root_state_v2(payloads[0])

        assert packet.strictly_valid
        assert packet.sequence == 1
        assert packet.estimate_time_ns == estimate_ns
        # The payload preserves the actual nearest source sample even though FK
        # uses the timestamp-interpolated joint state at the estimate time.
        assert packet.joint_time_ns == estimate_ns - 2_000_000
        assert packet.joint_sync_gap_ns == -2_000_000
        assert packet.correction_time_ns == estimate_ns - 1_000_000
        assert packet.calibration_digest == bridge._calibration_contract.digest
        assert packet.publish_time_ns >= packet.estimate_time_ns
        records = [
            json.loads(line)
            for line in replay_path.read_text(encoding="utf-8").splitlines()
        ]
        packet_records = [record for record in records if record.get("kind") == "packet"]
        assert len(packet_records) == 1
        assert bytes.fromhex(packet_records[0]["payload_hex"]) == payloads[0]
        assert packet_records[0]["joint_names"] == list(
            CANONICAL_G1_JOINT_NAMES
        )
        assert packet_records[0]["joint_position"] == pytest.approx(
            [0.0] * len(CANONICAL_G1_JOINT_NAMES)
        )
        assert packet_records[0]["joint_velocity"] == pytest.approx(
            [0.0] * len(CANONICAL_G1_JOINT_NAMES)
        )
        assert (
            packet_records[0]["joint_mapping_digest_sha256"]
            == canonical_joint_mapping_digest().hex()
        )
    finally:
        joint_sender.close()
        subscriber.close(linger=0)
        zmq_context.term()
        executor.remove_node(harness)
        executor.remove_node(bridge)
        harness.destroy_node()
        bridge.destroy_node()
        rclpy.shutdown()
