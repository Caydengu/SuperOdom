"""Typed ROS/UDP/ZMQ wrapper around the pure G1 pelvis bridge core."""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import socket
import time
from typing import Any

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Bool, String
from super_odometry_msgs.msg import (
    OptimizationStats,
    StateEstimationCalibration,
    StateEstimationCorrection,
)

from g1_root_state_bridge.bridge_core import (
    BridgeCore,
    BridgeCoreError,
    CorrectionSample,
    EstimatorCalibration,
    EstimatorHealthSample,
    EstimatorObservation,
    UNKNOWN_COVARIANCE_DIAGONAL,
)
from g1_root_state_bridge.calibration_contract import (
    CalibrationContract,
    load_installed_calibration_contract,
)
from g1_root_state_bridge.joint_transport import (
    JointPacketDecoder,
    JointPacketError,
    canonical_joint_mapping_digest,
)
from g1_root_state_bridge.joint_contract import TimedJointSample
from g1_root_state_bridge.kinematics import PelvisKinematics
from g1_root_state_bridge.protocol import (
    RootStatePacketV2,
    serialize_root_state_v2,
)


STATE_TOPIC = "/state_estimation"
STATS_TOPIC = "/super_odometry_stats"
HEALTH_TOPIC = "/state_estimation_health"
CALIBRATION_TOPIC = "/state_estimation_calibration"
CORRECTION_TOPIC = "/state_estimation_correction"
PELVIS_TOPIC = "/pelvis_state_estimation"
STATUS_TOPIC = "/pelvis_state_bridge/status"


def _stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _quaternion_xyzw_rotation(quaternion: Any) -> np.ndarray:
    values = np.asarray(
        (quaternion.x, quaternion.y, quaternion.z, quaternion.w),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise BridgeCoreError("ROS quaternion must be finite")
    norm = float(np.linalg.norm(values))
    if norm < 1e-12:
        raise BridgeCoreError("ROS quaternion has zero norm")
    x, y, z, w = values / norm
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        )
    )


def _pose_matrix(position: Any, orientation: Any) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quaternion_xyzw_rotation(orientation)
    transform[:3, 3] = (position.x, position.y, position.z)
    return transform


def _transform_matrix(transform_message: Any) -> np.ndarray:
    return _pose_matrix(transform_message.translation, transform_message.rotation)


def _observation_from_odometry(message: Odometry, receipt_time_ns: int) -> EstimatorObservation:
    return EstimatorObservation(
        estimate_time_ns=_stamp_ns(message.header.stamp),
        receipt_time_ns=receipt_time_ns,
        frame_id=message.header.frame_id,
        child_frame_id=message.child_frame_id,
        map_T_lidar=_pose_matrix(message.pose.pose.position, message.pose.pose.orientation),
        imu_linear_velocity_imu=np.asarray(
            (
                message.twist.twist.linear.x,
                message.twist.twist.linear.y,
                message.twist.twist.linear.z,
            ),
            dtype=np.float64,
        ),
        imu_angular_velocity_imu=np.asarray(
            (
                message.twist.twist.angular.x,
                message.twist.twist.angular.y,
                message.twist.twist.angular.z,
            ),
            dtype=np.float64,
        ),
        estimator_healthy=int(round(message.pose.covariance[0])) == 1,
    )


class G1RootStateBridgeNode(Node):
    """Join typed estimator/joint evidence and broadcast strict V2 packets."""

    def __init__(self, *, parameter_overrides: list[Any] | None = None) -> None:
        super().__init__(
            "g1_root_state_bridge",
            parameter_overrides=([] if parameter_overrides is None else parameter_overrides),
        )
        self.declare_parameter("joint_bind_host", "0.0.0.0")
        self.declare_parameter("joint_bind_port", 5576)
        self.declare_parameter("root_state_bind_endpoint", "tcp://*:5575")
        self.declare_parameter("calibration_digest_sha256", "")
        self.declare_parameter("max_joint_transport_age_ms", 10.0)
        self.declare_parameter("max_joint_sync_gap_ms", 10.0)
        self.declare_parameter("max_correction_input_age_ms", 150.0)
        self.declare_parameter("max_correction_age_ms", 250.0)
        self.declare_parameter("max_health_age_ms", 20.0)
        self.declare_parameter("pending_state_timeout_ms", 10.0)
        self.declare_parameter("replay_jsonl_path", "")

        self._calibration_contract: CalibrationContract = (
            load_installed_calibration_contract()
        )
        digest_hex = str(self.get_parameter("calibration_digest_sha256").value)
        configured_digest, digest_parameter_valid = self._parse_digest(digest_hex)
        self._calibration_digest = self._calibration_contract.digest
        self._digest_configured = (
            not digest_hex
            or (
                digest_parameter_valid
                and configured_digest == self._calibration_contract.digest
            )
        )
        self._process_epoch = time.time_ns() & ((1 << 63) - 1)
        self._last_calibration_reset_id: int | None = None
        self._core: BridgeCore | None = None
        self._pending: deque[EstimatorObservation] = deque(maxlen=64)
        self._latest_health_receipt_ns = 0
        self._last_stats_uncertainty: tuple[float, ...] | None = None
        self._pending_timeout_ns = int(
            float(self.get_parameter("pending_state_timeout_ms").value) * 1_000_000
        )

        joint_port = int(self.get_parameter("joint_bind_port").value)
        joint_host = str(self.get_parameter("joint_bind_host").value)
        if not 1 <= joint_port <= 65535:
            raise ValueError("joint_bind_port must be from 1 through 65535")
        self._joint_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._joint_socket.setblocking(False)
        self._joint_socket.bind((joint_host, joint_port))
        self._joint_decoder = JointPacketDecoder(
            allowed_mapping_digests={canonical_joint_mapping_digest()},
            max_age_ms=float(self.get_parameter("max_joint_transport_age_ms").value),
        )

        import zmq

        self._zmq = zmq
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.PUB)
        self._zmq_socket.setsockopt(zmq.LINGER, 0)
        self._zmq_socket.setsockopt(zmq.SNDHWM, 1)
        endpoint = str(self.get_parameter("root_state_bind_endpoint").value)
        self._zmq_socket.bind(endpoint)

        replay_path = str(self.get_parameter("replay_jsonl_path").value)
        self._replay_file = None
        if replay_path:
            path = Path(replay_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._replay_file = path.open("x", encoding="utf-8", buffering=1)

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
        self.create_subscription(Odometry, STATE_TOPIC, self._on_odometry, reliable)
        self.create_subscription(OptimizationStats, STATS_TOPIC, self._on_stats, reliable)
        self.create_subscription(
            StateEstimationCorrection,
            CORRECTION_TOPIC,
            self._on_correction,
            reliable,
        )
        self.create_subscription(Bool, HEALTH_TOPIC, self._on_health, reliable)
        self.create_subscription(
            StateEstimationCalibration,
            CALIBRATION_TOPIC,
            self._on_calibration,
            latched,
        )
        self._pelvis_publisher = self.create_publisher(Odometry, PELVIS_TOPIC, reliable)
        self._status_publisher = self.create_publisher(String, STATUS_TOPIC, reliable)
        self.create_timer(0.001, self._poll_joint_datagrams)
        self.get_logger().info(
            f"G1 root bridge listening for joints on {joint_host}:{joint_port} and publishing {endpoint}"
        )
        if not self._digest_configured:
            self.get_logger().error(
                "calibration_digest_sha256 does not match the locally computed contract; packets will fail calibration health"
            )

    @staticmethod
    def _parse_digest(value: str) -> tuple[bytes, bool]:
        try:
            digest = bytes.fromhex(value)
        except ValueError:
            return bytes(32), False
        return (digest, True) if len(digest) == 32 else (bytes(32), False)

    def _status(self, event: str, **fields: object) -> None:
        record = {
            "schema": "g1_root_state_bridge_status_v1",
            "event": event,
            "receipt_time_ns": time.time_ns(),
            **fields,
        }
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
        message = String()
        message.data = encoded
        self._status_publisher.publish(message)
        if self._replay_file is not None:
            self._replay_file.write(encoded + "\n")

    def _on_calibration(self, message: StateEstimationCalibration) -> None:
        reset_id = int(message.reset_id)
        if self._last_calibration_reset_id is not None:
            if reset_id == self._last_calibration_reset_id:
                return
            if reset_id < self._last_calibration_reset_id:
                self._status("calibration_rejected", reason="reset_id_not_increasing")
                return
        header_consistent = message.header.frame_id == message.map_frame
        imu_T_lidar = _transform_matrix(message.imu_t_lidar)
        static_extrinsic_consistent = np.allclose(
            imu_T_lidar,
            self._calibration_contract.imu_T_lidar,
            rtol=0.0,
            atol=1e-9,
        )
        calibration = EstimatorCalibration(
            semantics_version=message.semantics_version,
            map_frame=message.map_frame,
            sensor_frame=message.sensor_frame,
            gravity_frame=message.gravity_frame,
            initialization_time_ns=_stamp_ns(message.header.stamp),
            map_T_gravity=_transform_matrix(message.map_t_gravity),
            imu_T_lidar=imu_T_lidar,
            calibration_digest=self._calibration_digest,
            valid=bool(
                message.valid
                and header_consistent
                and static_extrinsic_consistent
                and self._digest_configured
            ),
        )
        source_epoch = self._process_epoch + reset_id
        try:
            self._core = BridgeCore(
                calibration=calibration,
                kinematics=PelvisKinematics(),
                source_epoch=source_epoch,
                allowed_calibration_digests=(
                    {self._calibration_digest} if self._digest_configured else set()
                ),
                max_joint_gap_ns=int(
                    float(self.get_parameter("max_joint_sync_gap_ms").value) * 1_000_000
                ),
                max_correction_input_age_ns=int(
                    float(
                        self.get_parameter("max_correction_input_age_ms").value
                    )
                    * 1_000_000
                ),
                max_correction_age_ns=int(
                    float(self.get_parameter("max_correction_age_ms").value) * 1_000_000
                ),
                max_health_age_ns=int(
                    float(self.get_parameter("max_health_age_ms").value) * 1_000_000
                ),
            )
        except (BridgeCoreError, RuntimeError, ValueError) as error:
            self._core = None
            self._status("calibration_rejected", reason=str(error))
            return
        self._last_calibration_reset_id = reset_id
        self._pending.clear()
        self._latest_health_receipt_ns = 0
        self._status(
            "calibration_accepted",
            reset_id=reset_id,
            source_epoch=source_epoch,
            calibration_valid=calibration.valid,
            calibration_digest_sha256=self._calibration_digest.hex(),
            static_extrinsic_consistent=static_extrinsic_consistent,
            calibration_manifest=self._calibration_contract.manifest,
        )

    def _on_stats(self, message: OptimizationStats) -> None:
        if self._core is None:
            return
        self._last_stats_uncertainty = (
            float(message.uncertainty_x),
            float(message.uncertainty_y),
            float(message.uncertainty_z),
            float(message.uncertainty_roll),
            float(message.uncertainty_pitch),
            float(message.uncertainty_yaw),
        )

    def _on_correction(self, message: StateEstimationCorrection) -> None:
        if self._core is None:
            return
        receipt_ns = time.time_ns()
        if message.semantics_version != "superodom-state-correction-v1":
            self._status("correction_rejected", reason="unknown_semantics_version")
            return
        if (
            self._last_calibration_reset_id is None
            or int(message.reset_id) != self._last_calibration_reset_id
        ):
            self._status("correction_rejected", reason="reset_id_mismatch")
            return
        if message.header.frame_id != self._core.calibration.map_frame:
            self._status("correction_rejected", reason="map_frame_mismatch")
            return
        if not bool(message.valid):
            self._status("correction_rejected", reason="estimator_did_not_apply")
            return
        evidence_time_ns = _stamp_ns(message.newest_observation_stamp)
        mapping_output_time_ns = _stamp_ns(message.mapping_output_stamp)
        application_time_ns = _stamp_ns(message.application_stamp)
        if not evidence_time_ns <= mapping_output_time_ns <= application_time_ns:
            self._status("correction_rejected", reason="invalid_mapping_timing_order")
            return
        try:
            # correction_time_ns in HSROOT02 is the newest physical LiDAR
            # return incorporated by an estimator-confirmed correction.  The
            # scan-start pose-reference and application clocks stay explicit
            # here instead of being conflated with freshness.
            sample = CorrectionSample(
                reference_time_ns=_stamp_ns(message.header.stamp),
                evidence_time_ns=evidence_time_ns,
                application_time_ns=application_time_ns,
                receipt_time_ns=receipt_ns,
                sequence=int(message.sequence),
                reset_id=int(message.reset_id),
                covariance_diagonal=UNKNOWN_COVARIANCE_DIAGONAL,
            )
            self._core.update_correction(sample)
        except BridgeCoreError as error:
            self._status("correction_rejected", reason=str(error))
            return
        self._status(
            "correction_accepted",
            sequence=sample.sequence,
            reset_id=sample.reset_id,
            reference_time_ns=sample.reference_time_ns,
            evidence_time_ns=sample.evidence_time_ns,
            mapping_output_time_ns=mapping_output_time_ns,
            application_time_ns=sample.application_time_ns,
        )
        self._drain_pending(receipt_ns)

    def _on_health(self, message: Bool) -> None:
        if self._core is None:
            return
        receipt_ns = time.time_ns()
        try:
            self._core.update_health(
                EstimatorHealthSample(receipt_time_ns=receipt_ns, healthy=bool(message.data))
            )
            self._latest_health_receipt_ns = receipt_ns
        except BridgeCoreError as error:
            self._status("health_rejected", reason=str(error))
        self._drain_pending(receipt_ns)

    def _on_odometry(self, message: Odometry) -> None:
        receipt_ns = time.time_ns()
        try:
            observation = _observation_from_odometry(message, receipt_ns)
        except (BridgeCoreError, ValueError) as error:
            self._status("estimate_rejected", reason=str(error))
            return
        if len(self._pending) == self._pending.maxlen:
            self._status("estimate_dropped", reason="pending_queue_overflow")
        self._pending.append(observation)
        self._drain_pending(receipt_ns)

    def _poll_joint_datagrams(self) -> None:
        for _ in range(256):
            try:
                payload, _source = self._joint_socket.recvfrom(2048)
            except BlockingIOError:
                break
            receipt_ns = time.time_ns()
            try:
                sample = self._joint_decoder.decode(payload, receipt_ns=receipt_ns)
            except JointPacketError as error:
                self._status(
                    "joint_rejected",
                    reason=self._joint_decoder.last_rejection_reason or str(error),
                )
                continue
            if self._core is not None:
                try:
                    self._core.append_joint(sample)
                except ValueError as error:
                    self._status("joint_rejected", reason=str(error))
        self._drain_pending(time.time_ns())

    def _drain_pending(self, now_ns: int) -> None:
        if self._core is None:
            return
        while self._pending:
            observation = self._pending[0]
            if self._latest_health_receipt_ns < observation.receipt_time_ns:
                if now_ns - observation.receipt_time_ns <= self._pending_timeout_ns:
                    return
                self._pending.popleft()
                self._status(
                    "estimate_dropped",
                    reason="health_sync_timeout",
                    estimate_time_ns=observation.estimate_time_ns,
                )
                continue
            try:
                result = self._core.build_packet_with_evidence(
                    observation,
                    publish_time_ns=now_ns,
                )
            except BridgeCoreError as error:
                if "joint synchronization failed" in str(error):
                    if now_ns - observation.receipt_time_ns <= self._pending_timeout_ns:
                        return
                    self._pending.popleft()
                    self._status(
                        "estimate_dropped",
                        reason="joint_sync_timeout",
                        estimate_time_ns=observation.estimate_time_ns,
                    )
                    continue
                self._pending.popleft()
                self._status(
                    "estimate_rejected",
                    reason=str(error),
                    estimate_time_ns=observation.estimate_time_ns,
                )
                continue
            self._pending.popleft()
            self._publish_packet(result.packet, result.joint_sample)

    def _publish_packet(
        self,
        packet: RootStatePacketV2,
        joint_sample: TimedJointSample,
    ) -> None:
        payload = serialize_root_state_v2(packet)
        try:
            self._zmq_socket.send(payload, flags=self._zmq.NOBLOCK)
        except self._zmq.Again:
            self._status("packet_dropped", reason="zmq_send_hwm", sequence=packet.sequence)
            return

        message = Odometry()
        message.header.stamp.sec = packet.estimate_time_ns // 1_000_000_000
        message.header.stamp.nanosec = packet.estimate_time_ns % 1_000_000_000
        message.header.frame_id = "gravity_aligned"
        message.child_frame_id = "pelvis"
        message.pose.pose.position.x, message.pose.pose.position.y, message.pose.pose.position.z = packet.position
        qw, qx, qy, qz = packet.quaternion_wxyz
        message.pose.pose.orientation.w = qw
        message.pose.pose.orientation.x = qx
        message.pose.pose.orientation.y = qy
        message.pose.pose.orientation.z = qz
        rotation_world_pelvis = _quaternion_xyzw_rotation(message.pose.pose.orientation)
        linear_pelvis = rotation_world_pelvis.T @ np.asarray(packet.linear_velocity)
        angular_pelvis = rotation_world_pelvis.T @ np.asarray(packet.angular_velocity)
        (
            message.twist.twist.linear.x,
            message.twist.twist.linear.y,
            message.twist.twist.linear.z,
        ) = tuple(float(value) for value in linear_pelvis)
        (
            message.twist.twist.angular.x,
            message.twist.twist.angular.y,
            message.twist.twist.angular.z,
        ) = tuple(float(value) for value in angular_pelvis)
        for index, value in zip((0, 7, 14, 21, 28, 35), packet.covariance_diagonal, strict=True):
            message.pose.covariance[index] = value
        self._pelvis_publisher.publish(message)
        self._status(
            "packet_published",
            kind="packet",
            payload_hex=payload.hex(),
            sequence=packet.sequence,
            source_epoch=packet.source_epoch,
            estimate_time_ns=packet.estimate_time_ns,
            publish_time_ns=packet.publish_time_ns,
            correction_time_ns=packet.correction_time_ns,
            joint_time_ns=packet.joint_time_ns,
            joint_sync_gap_ns=packet.joint_sync_gap_ns,
            joint_names=joint_sample.names,
            joint_position=joint_sample.position,
            joint_velocity=joint_sample.velocity,
            joint_mapping_digest_sha256=(
                canonical_joint_mapping_digest().hex()
            ),
            health_flags=int(packet.health_flags),
            strictly_valid=packet.strictly_valid,
            position=packet.position,
            quaternion_wxyz=packet.quaternion_wxyz,
            linear_velocity_world=packet.linear_velocity,
            angular_velocity_world=packet.angular_velocity,
            estimator_uncertainty_scores=self._last_stats_uncertainty,
            calibration_digest_sha256=packet.calibration_digest.hex(),
        )

    def destroy_node(self) -> bool:
        if self._replay_file is not None:
            self._replay_file.close()
            self._replay_file = None
        self._joint_socket.close()
        self._zmq_socket.close(linger=0)
        self._zmq_context.term()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = G1RootStateBridgeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
