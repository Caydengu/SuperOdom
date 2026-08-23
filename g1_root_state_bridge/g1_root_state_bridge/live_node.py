"""Command-incapable ROS 2 and HSROOT02 producer for selected G1 localization."""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from typing import Any

import numpy as np

from g1_root_state_bridge.clock_sync import (
    ClockMapConfig,
    ClockSampleDisposition,
    ClockSynchronizationError,
    OnlineAffineClockMapper,
    classify_clock_estimate,
)
from g1_root_state_bridge.joint_contract import CANONICAL_G1_JOINT_NAMES, TimedJointSample
from g1_root_state_bridge.kiss_registration import KissRegistration
from g1_root_state_bridge.live_pipeline import (
    LiveLocalizationConfig,
    LiveLocalizationError,
    SelectedLocalizationPipeline,
)
from g1_root_state_bridge.pointcloud2_adapter import decode_livox_pointcloud2, header_time_ns
from g1_root_state_bridge.udp_lowstate import DynamicLowStateReceiver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lidar-topic", default="/utlidar/cloud_livox_mid360")
    parser.add_argument("--imu-topic", default="/utlidar/imu_livox_mid360")
    parser.add_argument("--odom-topic", default="/g1/localization/pelvis_odom")
    parser.add_argument(
        "--registered-cloud-topic",
        default="/g1/localization/cloud_registered",
    )
    parser.add_argument("--lowstate-bind-host", default="0.0.0.0")
    parser.add_argument("--lowstate-bind-port", type=int, default=5589)
    parser.add_argument("--zmq-bind", default="tcp://*:5575")
    parser.add_argument("--gyro-bias-radps", nargs=3, type=float, required=True)
    parser.add_argument("--voxel-size-m", type=float, default=0.15)
    parser.add_argument("--minimum-range-m", type=float, default=0.5)
    parser.add_argument("--maximum-range-m", type=float, default=15.0)
    parser.add_argument("--maximum-points", type=int, default=5_000)
    parser.add_argument(
        "--point-time-unit",
        choices=("nanoseconds", "seconds"),
        default="nanoseconds",
    )
    parser.add_argument("--clock-minimum-samples", type=int, default=8)
    parser.add_argument("--clock-maximum-residual-p95-ms", type=float, default=5.0)
    parser.add_argument("--clock-maximum-transport-delay-ms", type=float, default=10.0)
    return parser.parse_args()


def run_node(args: argparse.Namespace) -> None:
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Imu, PointCloud2, PointField
    import zmq

    class G1KissLocalizationNode(Node):
        def __init__(self) -> None:
            super().__init__("g1_kiss_live_localization")
            clock_config = ClockMapConfig(
                minimum_samples=args.clock_minimum_samples,
                maximum_residual_p95_ns=round(
                    args.clock_maximum_residual_p95_ms * 1e6
                ),
                maximum_transport_delay_ns=round(
                    args.clock_maximum_transport_delay_ms * 1e6
                ),
            )
            self.imu_clock = OnlineAffineClockMapper(clock_config)
            self.lowstate_clock = OnlineAffineClockMapper(clock_config)
            self.pipeline = SelectedLocalizationPipeline(
                KissRegistration(
                    voxel_size_m=args.voxel_size_m,
                    minimum_range_m=args.minimum_range_m,
                    maximum_range_m=args.maximum_range_m,
                ),
                LiveLocalizationConfig(gyro_bias_radps=tuple(args.gyro_bias_radps)),
            )
            self.lock = threading.RLock()
            self.active = False
            self.pipeline_epoch = 0
            self.lowstate_source_epoch: int | None = None
            self.last_lidar_source_time_ns: int | None = None
            self.scan_queue: queue.Queue[tuple[int, np.ndarray, np.ndarray, int]] = queue.Queue(
                maxsize=1
            )
            self.stop_event = threading.Event()
            self.worker = threading.Thread(target=self._worker, name="g1-kiss-worker", daemon=True)
            self.stats: dict[str, int] = {
                "imu": 0,
                "lowstate": 0,
                "lidar": 0,
                "published": 0,
                "cloud_published": 0,
                "queue_replaced": 0,
                "clock_rejected": 0,
                "pipeline_rejected": 0,
                "transport_rejected": 0,
                "epoch_discarded": 0,
            }
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.create_subscription(Imu, args.imu_topic, self._imu_callback, qos)
            self.create_subscription(PointCloud2, args.lidar_topic, self._lidar_callback, qos)
            self.odom_publisher = self.create_publisher(Odometry, args.odom_topic, 1)
            self.cloud_publisher = self.create_publisher(
                PointCloud2, args.registered_cloud_topic, 1
            )
            self.zmq_context = zmq.Context.instance()
            self.zmq_socket = self.zmq_context.socket(zmq.PUB)
            self.zmq_socket.setsockopt(zmq.SNDHWM, 1)
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.bind(args.zmq_bind)
            self.lowstate_receiver = DynamicLowStateReceiver(
                bind_host=args.lowstate_bind_host,
                bind_port=args.lowstate_bind_port,
                callback=self._lowstate_callback,
            )
            self.lowstate_receiver.start()
            self.worker.start()
            self.create_timer(5.0, self._report)

        def _activate_if_ready(self) -> None:
            if self.active or not (self.imu_clock.valid and self.lowstate_clock.valid):
                return
            self.pipeline_epoch += 1
            self.pipeline.reset(self.pipeline_epoch)
            self.active = True
            self.get_logger().info(
                json.dumps(
                    {
                        "event": "source_clocks_admitted",
                        "pipeline_epoch": self.pipeline_epoch,
                        "imu_residual_p95_ms": self.imu_clock.estimate(
                            self.imu_clock.last_source_time_ns
                        ).residual_p95_ns
                        * 1e-6,
                        "imu_transport_delay_ms": self.imu_clock.estimate(
                            self.imu_clock.last_source_time_ns
                        ).transport_delay_ns
                        * 1e-6,
                        "lowstate_residual_p95_ms": self.lowstate_clock.estimate(
                            self.lowstate_clock.last_source_time_ns
                        ).residual_p95_ns
                        * 1e-6,
                        "lowstate_transport_delay_ms": self.lowstate_clock.estimate(
                            self.lowstate_clock.last_source_time_ns
                        ).transport_delay_ns
                        * 1e-6,
                    },
                    sort_keys=True,
                )
            )

        def _deactivate(self, reason: str) -> None:
            if self.active:
                self.get_logger().error(json.dumps({"event": "localization_deactivated", "reason": reason}))
            self.active = False

        def _observe_clock(
            self,
            mapper: OnlineAffineClockMapper,
            source_ns: int,
            receipt_ns: int,
            *,
            stream: str,
        ) -> Any | None:
            """Observe one clock pair, resetting the lane on host-time anomalies."""

            try:
                return mapper.observe(source_ns, receipt_ns)
            except ClockSynchronizationError as error:
                mapper.reset()
                self._deactivate(f"{stream}_clock_contract:{error}")
                self.stats["clock_rejected"] += 1
                return None

        def _imu_callback(self, message: Any) -> None:
            receipt_ns = time.time_ns()
            try:
                source_ns = header_time_ns(message)
            except ValueError:
                self.stats["clock_rejected"] += 1
                self._deactivate("imu_header_time_invalid")
                return
            with self.lock:
                previous_epoch = self.imu_clock.source_epoch
                estimate = self._observe_clock(
                    self.imu_clock, source_ns, receipt_ns, stream="imu"
                )
                if estimate is None:
                    return
                if estimate.source_epoch != previous_epoch:
                    self._deactivate("imu_clock_epoch_changed")
                disposition = classify_clock_estimate(estimate, lane_active=self.active)
                if disposition is ClockSampleDisposition.RESET_LANE:
                    self._deactivate(
                        f"imu_clock_fit_unhealthy:{','.join(estimate.health_reasons)}"
                    )
                self._activate_if_ready()
                if not self.active or disposition is not ClockSampleDisposition.ACCEPT:
                    self.stats["clock_rejected"] += 1
                    return
                angular = np.asarray(
                    (
                        message.angular_velocity.x,
                        message.angular_velocity.y,
                        message.angular_velocity.z,
                    ),
                    dtype=np.float64,
                )
                try:
                    self.pipeline.append_imu(estimate.mapped_time_ns, angular)
                    self.stats["imu"] += 1
                except LiveLocalizationError:
                    self._deactivate("imu_pipeline_contract")

        def _lowstate_callback(self, packet: Any, receipt_ns: int) -> None:
            with self.lock:
                if self.lowstate_source_epoch is None:
                    self.lowstate_source_epoch = int(packet.source_epoch)
                elif int(packet.source_epoch) != self.lowstate_source_epoch:
                    self.lowstate_source_epoch = int(packet.source_epoch)
                    self.lowstate_clock.reset()
                    self._deactivate("lowstate_source_epoch_changed")
                previous_epoch = self.lowstate_clock.source_epoch
                estimate = self._observe_clock(
                    self.lowstate_clock,
                    int(packet.robot_stamp_ns),
                    receipt_ns,
                    stream="lowstate",
                )
                if estimate is None:
                    return
                if estimate.source_epoch != previous_epoch:
                    self._deactivate("lowstate_clock_epoch_changed")
                disposition = classify_clock_estimate(estimate, lane_active=self.active)
                if disposition is ClockSampleDisposition.RESET_LANE:
                    self._deactivate(
                        f"lowstate_clock_fit_unhealthy:{','.join(estimate.health_reasons)}"
                    )
                self._activate_if_ready()
                if not self.active or disposition is not ClockSampleDisposition.ACCEPT:
                    self.stats["clock_rejected"] += 1
                    return
                sample = TimedJointSample(
                    stamp_ns=estimate.mapped_time_ns,
                    receipt_ns=max(receipt_ns, estimate.mapped_time_ns),
                    names=CANONICAL_G1_JOINT_NAMES,
                    position=tuple(packet.joint_position),
                    velocity=tuple(packet.joint_velocity),
                    sequence=int(packet.sequence),
                    source_epoch=self.pipeline_epoch,
                )
                try:
                    self.pipeline.append_joint(sample)
                    self.stats["lowstate"] += 1
                except ValueError:
                    self._deactivate("lowstate_pipeline_contract")

        def _lidar_callback(self, message: Any) -> None:
            try:
                source_start_ns = header_time_ns(message)
            except ValueError:
                self.stats["clock_rejected"] += 1
                self._deactivate("lidar_header_time_invalid")
                return
            with self.lock:
                if self.last_lidar_source_time_ns is not None and source_start_ns <= self.last_lidar_source_time_ns:
                    self._deactivate("lidar_source_time_regressed")
                    self.last_lidar_source_time_ns = source_start_ns
                    return
                self.last_lidar_source_time_ns = source_start_ns
                if not self.active or not self.imu_clock.valid:
                    self.stats["clock_rejected"] += 1
                    return
                mapped_start_ns = self.imu_clock.estimate(source_start_ns).mapped_time_ns
                queued_epoch = self.pipeline_epoch
            try:
                points, relative = decode_livox_pointcloud2(
                    message,
                    minimum_range_m=args.minimum_range_m,
                    maximum_range_m=args.maximum_range_m,
                    maximum_points=args.maximum_points,
                    time_unit=args.point_time_unit,
                )
            except ValueError:
                self.stats["pipeline_rejected"] += 1
                return
            item = (queued_epoch, points, relative, mapped_start_ns)
            try:
                self.scan_queue.put_nowait(item)
            except queue.Full:
                try:
                    self.scan_queue.get_nowait()
                except queue.Empty:
                    pass
                self.scan_queue.put_nowait(item)
                self.stats["queue_replaced"] += 1
            self.stats["lidar"] += 1

        def _worker(self) -> None:
            while not self.stop_event.is_set():
                try:
                    queued_epoch, points, relative, start_ns = self.scan_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                with self.lock:
                    if not self.active or queued_epoch != self.pipeline_epoch:
                        self.stats["epoch_discarded"] += 1
                        continue
                    try:
                        output = self.pipeline.process_scan(
                            points,
                            relative,
                            scan_start_time_ns=start_ns,
                            clock_valid=self.imu_clock.valid and self.lowstate_clock.valid,
                        )
                    except LiveLocalizationError:
                        self.stats["pipeline_rejected"] += 1
                        continue
                try:
                    self.zmq_socket.send(output.payload, flags=zmq.NOBLOCK)
                except zmq.Again:
                    self.stats["transport_rejected"] += 1
                packet = output.packet
                message = Odometry()
                message.header.stamp.sec = packet.estimate_time_ns // 1_000_000_000
                message.header.stamp.nanosec = packet.estimate_time_ns % 1_000_000_000
                message.header.frame_id = "kiss_local"
                message.child_frame_id = "pelvis_navigation_yaw"
                message.pose.pose.position.x, message.pose.pose.position.y, message.pose.pose.position.z = packet.position
                w, x, y, z = packet.quaternion_wxyz
                message.pose.pose.orientation.w = w
                message.pose.pose.orientation.x = x
                message.pose.pose.orientation.y = y
                message.pose.pose.orientation.z = z
                message.twist.twist.linear.x, message.twist.twist.linear.y, message.twist.twist.linear.z = packet.linear_velocity
                message.twist.twist.angular.x, message.twist.twist.angular.y, message.twist.twist.angular.z = packet.angular_velocity
                self.odom_publisher.publish(message)
                cloud = PointCloud2()
                cloud.header.stamp.sec = packet.estimate_time_ns // 1_000_000_000
                cloud.header.stamp.nanosec = packet.estimate_time_ns % 1_000_000_000
                cloud.header.frame_id = "kiss_local"
                cloud.height = 1
                cloud.width = int(output.registered_points_local_xyz_m.shape[0])
                cloud.fields = [
                    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                ]
                cloud.is_bigendian = False
                cloud.point_step = 12
                cloud.row_step = cloud.point_step * cloud.width
                cloud.data = output.registered_cloud_xyz32
                cloud.is_dense = True
                self.cloud_publisher.publish(cloud)
                self.stats["published"] += 1
                self.stats["cloud_published"] += 1

        def _report(self) -> None:
            self.get_logger().info(
                json.dumps(
                    {
                        "schema": "g1_kiss_live_localization_status_v1",
                        "active": self.active,
                        "pipeline_epoch": self.pipeline_epoch,
                        "stats": self.stats,
                        "lowstate_udp_invalid": self.lowstate_receiver.invalid,
                        "command_capability": "structurally_unavailable",
                    },
                    sort_keys=True,
                )
            )

        def close(self) -> None:
            self.stop_event.set()
            self.worker.join(timeout=2.0)
            self.lowstate_receiver.close()
            self.zmq_socket.close()

    rclpy.init()
    node = G1KissLocalizationNode()
    try:
        rclpy.spin(node)
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    run_node(parse_args())


if __name__ == "__main__":
    main()
