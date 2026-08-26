"""ROS 2 and RVMAP001 producer for automatic Polycam map localization.

The node is command-incapable: it subscribes only to the selected KISS-local
registered cloud and pelvis odometry, observes HSROOT02 health over a SUB
socket, and publishes accepted slow-lane corrections over a PUB socket.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from g1_root_state_bridge.map_protocol import serialize_map_correction_v1
from g1_root_state_bridge.pointcloud2_adapter import (
    decode_xyz_pointcloud2,
    header_time_ns,
)
from g1_root_state_bridge.protocol import (
    REQUIRED_ROOT_FUSION_FLAGS,
    RootStatePacketV2,
    deserialize_root_state_v2,
)
from g1_root_state_bridge.structural_map_localization import (
    AutomaticMapCorrectionEngine,
    MapCorrectionAttempt,
    RegisteredMapEvidenceBuffer,
    StructuralMapConfig,
    StructuralMapLocalizationError,
    TimedLocalPose,
    TimedRegisteredCloud,
)
from g1_root_state_bridge.ui_initialization import (
    UIInitializationError,
    load_ui_map_initialization,
)


@dataclass(frozen=True)
class RootEvidenceSnapshot:
    packet: RootStatePacketV2
    receipt_time_ns: int


def root_evidence_rejection_reason(
    snapshot: RootEvidenceSnapshot | None,
    *,
    evidence_time_ns: int,
    now_ns: int,
    config: StructuralMapConfig,
) -> str | None:
    if snapshot is None:
        return "root_state_unavailable"
    packet = snapshot.packet
    if (packet.health_flags & REQUIRED_ROOT_FUSION_FLAGS) != REQUIRED_ROOT_FUSION_FLAGS:
        return "root_state_health"
    if evidence_time_ns <= 0 or evidence_time_ns > now_ns:
        return "evidence_clock"
    if packet.estimate_time_ns < evidence_time_ns:
        return "root_state_not_caught_up"
    if packet.estimate_time_ns - evidence_time_ns > config.maximum_root_evidence_gap_ns:
        return "root_evidence_gap"
    if now_ns - evidence_time_ns > config.maximum_input_age_ns:
        return "map_evidence_stale_before_registration"
    return None


def ui_root_evidence_rejection_reason(
    snapshot: RootEvidenceSnapshot | None,
    *,
    evidence_time_ns: int,
    receipt_created_ns: int,
    now_ns: int,
    config: StructuralMapConfig,
) -> str | None:
    """Validate a bounded delayed UI result without weakening live-map gates."""

    if snapshot is None:
        return "root_state_unavailable"
    packet = snapshot.packet
    if (packet.health_flags & REQUIRED_ROOT_FUSION_FLAGS) != REQUIRED_ROOT_FUSION_FLAGS:
        return "root_state_health"
    if evidence_time_ns <= 0 or evidence_time_ns > receipt_created_ns:
        return "evidence_clock"
    if receipt_created_ns > now_ns:
        return "receipt_clock"
    if receipt_created_ns - evidence_time_ns > config.maximum_ui_initialization_age_ns:
        return "ui_initialization_latency"
    if now_ns - receipt_created_ns > config.maximum_ui_receipt_age_ns:
        return "ui_receipt_stale"
    if packet.estimate_time_ns < evidence_time_ns:
        return "root_state_not_caught_up"
    return None


class RootStateMonitor:
    """Latest-only HSROOT02 subscriber with no command or service capability."""

    def __init__(self, endpoint: str) -> None:
        import zmq

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._snapshot: RootEvidenceSnapshot | None = None
        self.invalid_packets = 0
        self._thread = threading.Thread(
            target=self._run,
            name="g1-map-root-monitor",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        import zmq

        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while not self._stop.is_set():
            try:
                if self._socket not in dict(poller.poll(50)):
                    continue
                payload = self._socket.recv()
                try:
                    packet = deserialize_root_state_v2(payload)
                except ValueError:
                    self.invalid_packets += 1
                    continue
                with self._lock:
                    self._snapshot = RootEvidenceSnapshot(packet, time.time_ns())
            except zmq.ZMQError:
                if not self._stop.is_set():
                    self.invalid_packets += 1
                    self._stop.wait(0.05)

    def snapshot(self) -> RootEvidenceSnapshot | None:
        with self._lock:
            return self._snapshot

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._socket.close(linger=0)
        self._context.term()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--map-key", default="map_xy_all_5cm")
    parser.add_argument("--map-version", type=int, default=1)
    parser.add_argument("--map-epoch", type=int, default=1)
    parser.add_argument(
        "--registered-cloud-topic",
        default="/g1/localization/cloud_registered",
    )
    parser.add_argument("--odom-topic", default="/g1/localization/pelvis_odom")
    parser.add_argument("--root-endpoint", default="tcp://127.0.0.1:5575")
    parser.add_argument("--map-bind", default="tcp://*:5577")
    parser.add_argument("--maximum-points-per-cloud", type=int, default=5_000)
    parser.add_argument("--global-window-sec", type=float, default=10.0)
    parser.add_argument("--tracking-window-sec", type=float, default=5.0)
    parser.add_argument("--tracking-period-sec", type=float, default=2.0)
    parser.add_argument(
        "--initialization-mode", choices=("automatic", "ui"), default="automatic"
    )
    parser.add_argument("--ui-initialization-receipt", type=Path)
    parser.add_argument("--expected-glb-sha256")
    parser.add_argument("--expected-surface-sha256")
    args = parser.parse_args()
    if args.initialization_mode == "ui" and (
        args.ui_initialization_receipt is None
        or args.expected_glb_sha256 is None
        or args.expected_surface_sha256 is None
    ):
        parser.error(
            "ui initialization requires --ui-initialization-receipt, "
            "--expected-glb-sha256, and --expected-surface-sha256"
        )
    return args


def run_node(args: argparse.Namespace) -> None:
    import rclpy
    import zmq
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from sensor_msgs.msg import PointCloud2

    class G1StructuralMapNode(Node):
        def __init__(self) -> None:
            super().__init__("g1_structural_map_localization")
            self.config = StructuralMapConfig(
                global_window_sec=args.global_window_sec,
                tracking_window_sec=args.tracking_window_sec,
                tracking_period_sec=args.tracking_period_sec,
            )
            self.engine = AutomaticMapCorrectionEngine(
                map_path=args.map,
                map_key=args.map_key,
                map_version=args.map_version,
                map_epoch=args.map_epoch,
                config=self.config,
            )
            self.buffer = RegisteredMapEvidenceBuffer(
                maximum_history_sec=max(15.0, args.global_window_sec + 2.0)
            )
            self.root = RootStateMonitor(args.root_endpoint)
            # rclpy.node.Node already exposes a read-only ``context`` property.
            # Keep the ZeroMQ lifetime under a distinct name so constructing the
            # ROS node does not overwrite that base-class property.
            self.zmq_context = zmq.Context()
            self.publisher = self.zmq_context.socket(zmq.PUB)
            self.publisher.setsockopt(zmq.SNDHWM, 1)
            self.publisher.setsockopt(zmq.LINGER, 0)
            self.publisher.bind(args.map_bind)
            self.lock = threading.RLock()
            self.work: queue.Queue[int] = queue.Queue(maxsize=1)
            self.stop_event = threading.Event()
            self.last_attempt_evidence_ns = 0
            self.bound_epoch: int | None = None
            self.stats: Counter[str] = Counter()
            self.attempt_runtime_ms: deque[float] = deque(maxlen=2_000)
            self.last_attempt: dict[str, object] | None = None
            self.last_ui_content_sha256: str | None = None
            self.last_ui_file_signature: tuple[int, int] | None = None
            qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=4,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.create_subscription(
                PointCloud2,
                args.registered_cloud_topic,
                self._cloud_callback,
                qos,
            )
            self.create_subscription(
                Odometry, args.odom_topic, self._odom_callback, qos
            )
            self.worker = threading.Thread(
                target=self._worker,
                name="g1-structural-map-worker",
                daemon=True,
            )
            self.worker.start()
            if args.initialization_mode == "ui":
                self.create_timer(0.1, self._try_ui_initialization)
            self.create_timer(5.0, self._report)

        def _record_rejection(self, reason: str) -> None:
            self.stats[f"rejected:{reason}"] += 1

        def _cloud_callback(self, message: Any) -> None:
            try:
                source_ns = header_time_ns(message)
                points = decode_xyz_pointcloud2(
                    message,
                    maximum_points=args.maximum_points_per_cloud,
                )
                row = TimedRegisteredCloud(
                    source_ns,
                    str(message.header.frame_id),
                    points,
                )
                with self.lock:
                    self.buffer.append_cloud(row)
                self.stats["clouds"] += 1
                try:
                    self.work.put_nowait(source_ns)
                except queue.Full:
                    try:
                        self.work.get_nowait()
                    except queue.Empty:
                        pass
                    self.work.put_nowait(source_ns)
                    self.stats["work_replaced"] += 1
            except ValueError as error:
                self._record_rejection(f"cloud:{error}")

        def _odom_callback(self, message: Any) -> None:
            try:
                source_ns = header_time_ns(message)
                row = TimedLocalPose(
                    source_ns,
                    str(message.header.frame_id),
                    np.asarray(
                        (message.pose.pose.position.x, message.pose.pose.position.y),
                        dtype=np.float64,
                    ),
                )
                with self.lock:
                    self.buffer.append_pose(row)
                self.stats["poses"] += 1
            except ValueError as error:
                self._record_rejection(f"pose:{error}")

        def _attempt_summary(self, attempt: MapCorrectionAttempt) -> dict[str, object]:
            packet = attempt.packet
            icp = attempt.report.get("icp", attempt.report)
            observability = icp.get("observability", {})
            return {
                "schema": "g1_structural_map_attempt_v1",
                "kind": attempt.kind,
                "accepted": attempt.accepted,
                "rejection_reason": attempt.rejection_reason,
                "reference_time_ns": attempt.reference_time_ns,
                "evidence_time_ns": attempt.evidence_time_ns,
                "runtime_ms": attempt.runtime_ms,
                "sequence": None if packet is None else packet.sequence,
                "fitness": icp.get("inlier_fraction"),
                "rmse_m": icp.get("rmse_m"),
                "p95_m": icp.get("p95_m"),
                "min_eig": observability.get("minimum_eigenvalue"),
                "condition_number": observability.get("condition_number"),
                "pose_jump_m": icp.get("pose_jump_m"),
                "yaw_jump_deg": icp.get("yaw_jump_deg"),
            }

        def _publish_attempt(self, attempt: MapCorrectionAttempt) -> None:
            self.attempt_runtime_ms.append(attempt.runtime_ms)
            self.last_attempt = self._attempt_summary(attempt)
            self.get_logger().info(json.dumps(self.last_attempt, sort_keys=True))
            if not attempt.accepted or attempt.packet is None:
                self._record_rejection(attempt.rejection_reason or "map_gate")
                return
            try:
                self.publisher.send(
                    serialize_map_correction_v1(attempt.packet),
                    flags=zmq.NOBLOCK,
                )
                self.stats["packets_published"] += 1
            except zmq.Again:
                self._record_rejection("publisher_backpressure")

        def _try_ui_initialization(self) -> None:
            receipt_path = args.ui_initialization_receipt
            if receipt_path is None:
                return
            try:
                stat = receipt_path.stat()
            except FileNotFoundError:
                return
            signature = (int(stat.st_mtime_ns), int(stat.st_size))
            if signature == self.last_ui_file_signature:
                return
            try:
                receipt = load_ui_map_initialization(
                    receipt_path,
                    expected_structural_map_sha256=self.engine.map_digest.hex(),
                    expected_glb_sha256=args.expected_glb_sha256,
                    expected_surface_sha256=args.expected_surface_sha256,
                    minimum_fitness=self.config.minimum_inlier_fraction,
                    maximum_rmse_m=self.config.maximum_rmse_m,
                    minimum_eigenvalue=self.config.minimum_observability_eigenvalue,
                    maximum_condition_number=self.config.maximum_observability_condition_number,
                    maximum_correction_m=0.75,
                    maximum_correction_yaw_deg=20.0,
                )
            except (OSError, ValueError, UIInitializationError) as error:
                self.last_ui_file_signature = signature
                self._record_rejection(f"ui_receipt:{error}")
                return
            if receipt.content_sha256 == self.last_ui_content_sha256:
                return
            snapshot = self.root.snapshot()
            now_ns = time.time_ns()
            reason = ui_root_evidence_rejection_reason(
                snapshot,
                evidence_time_ns=receipt.evidence_time_ns,
                receipt_created_ns=receipt.created_realtime_ns,
                now_ns=now_ns,
                config=self.config,
            )
            if reason is not None:
                if reason in {
                    "evidence_clock",
                    "receipt_clock",
                    "ui_initialization_latency",
                    "ui_receipt_stale",
                }:
                    self.last_ui_file_signature = signature
                self._record_rejection(f"ui_receipt:{reason}")
                return
            assert snapshot is not None
            try:
                # Serialize explicit re-anchoring with background tracking so a
                # single RVMAP001 sequence always names one complete map anchor.
                with self.lock:
                    if (
                        self.bound_epoch is None
                        or snapshot.packet.source_epoch != self.bound_epoch
                    ):
                        self._record_rejection("ui_receipt:local_epoch_unbound")
                        return
                    relocalizing = self.engine.rotation is not None
                    attempt = self.engine.initialize_from_ui(
                        map_T_local=receipt.map_T_local,
                        fitness=receipt.fitness,
                        rmse_m=receipt.rmse_m,
                        min_eig=receipt.min_eig,
                        cond_number=receipt.cond_number,
                        reference_time_ns=receipt.reference_time_ns,
                        evidence_time_ns=receipt.evidence_time_ns,
                        application_time_ns=now_ns,
                    )
                    self.last_ui_content_sha256 = receipt.content_sha256
                    self.last_ui_file_signature = signature
                    self.stats["ui_receipts_accepted"] += 1
                    if relocalizing:
                        self.stats["ui_relocalizations_accepted"] += 1
            except StructuralMapLocalizationError as error:
                self._record_rejection(f"ui_receipt:{error}")
                return
            self._publish_attempt(attempt)

        def _worker(self) -> None:
            while not self.stop_event.is_set():
                try:
                    evidence_ns = self.work.get(timeout=0.2)
                except queue.Empty:
                    continue
                snapshot = self.root.snapshot()
                now_ns = time.time_ns()
                reason = root_evidence_rejection_reason(
                    snapshot,
                    evidence_time_ns=evidence_ns,
                    now_ns=now_ns,
                    config=self.config,
                )
                if reason is not None:
                    # A caught-up root packet may arrive immediately after the cloud.
                    if reason == "root_state_not_caught_up":
                        self.stop_event.wait(0.02)
                        try:
                            self.work.put_nowait(evidence_ns)
                        except queue.Full:
                            pass
                    else:
                        self._record_rejection(reason)
                    continue
                assert snapshot is not None
                source_epoch = snapshot.packet.source_epoch
                with self.lock:
                    if self.bound_epoch != source_epoch:
                        self.bound_epoch = source_epoch
                        self.engine.bind_local_epoch(source_epoch)
                        self.buffer.clear()
                        self.last_attempt_evidence_ns = 0
                        self.stats["epoch_resets"] += 1
                        continue
                    if args.initialization_mode == "ui" and self.engine.rotation is None:
                        continue
                    if self.engine.rotation is None:
                        window_sec = self.config.global_window_sec
                    else:
                        period_ns = round(self.config.tracking_period_sec * 1e9)
                        if evidence_ns - self.last_attempt_evidence_ns < period_ns:
                            continue
                        window_sec = self.config.tracking_window_sec
                    try:
                        query, audit = self.buffer.structural_query(
                            end_source_time_ns=evidence_ns,
                            window_sec=window_sec,
                            config=self.config,
                        )
                        if self.engine.rotation is None:
                            attempt = self.engine.initialize(
                                query,
                                reference_time_ns=int(audit["start_source_time_ns"]),
                                evidence_time_ns=evidence_ns,
                            )
                        else:
                            pose = self.buffer.pose_at(
                                evidence_ns,
                                maximum_age_ns=self.config.maximum_root_evidence_gap_ns,
                            )
                            attempt = self.engine.track(
                                query,
                                local_position_xy_m=pose.position_xy_m,
                                reference_time_ns=int(audit["start_source_time_ns"]),
                                evidence_time_ns=evidence_ns,
                            )
                    except StructuralMapLocalizationError as error:
                        self._record_rejection(f"query:{error}")
                        continue
                    self.last_attempt_evidence_ns = evidence_ns
                self._publish_attempt(attempt)

        def _report(self) -> None:
            runtimes = np.asarray(self.attempt_runtime_ms, dtype=np.float64)
            runtime = (
                {}
                if not runtimes.size
                else {
                    "p50": float(np.quantile(runtimes, 0.50)),
                    "p95": float(np.quantile(runtimes, 0.95)),
                    "p99": float(np.quantile(runtimes, 0.99)),
                    "maximum": float(np.max(runtimes)),
                }
            )
            self.get_logger().info(
                json.dumps(
                    {
                        "schema": "g1_structural_map_localization_status_v1",
                        "command_capability": "structurally_unavailable",
                        "map_digest": self.engine.map_digest.hex(),
                        "map_key": self.engine.map_key,
                        "initialization_mode": args.initialization_mode,
                        "ui_receipt_content_sha256": self.last_ui_content_sha256,
                        "local_source_epoch": self.bound_epoch,
                        "initialized": self.engine.rotation is not None,
                        "stats": dict(self.stats),
                        "attempt_runtime_ms": runtime,
                        "invalid_root_packets": self.root.invalid_packets,
                        "last_attempt": self.last_attempt,
                    },
                    sort_keys=True,
                )
            )

        def close(self) -> None:
            self.stop_event.set()
            self.worker.join(timeout=2.0)
            self.root.close()
            self.publisher.close(linger=0)
            self.zmq_context.term()

    rclpy.init()
    node = G1StructuralMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    run_node(parse_args())


if __name__ == "__main__":
    main()
