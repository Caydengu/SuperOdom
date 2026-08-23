"""Subscriber-only Unitree G1 low-state and IMU capture relay."""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import threading
import time
from typing import Any

from g1_root_state_bridge.dynamic_capture_transport import (
    REQUIRED_DYNAMIC_CAPTURE_HEALTH,
    DynamicCapturePacketV1,
    serialize_dynamic_capture_packet,
)
from g1_root_state_bridge.joint_transport import canonical_joint_mapping_digest


class G1DynamicCaptureRelay:
    """Read typed low state and forward fixed, CRC-protected UDP datagrams."""

    def __init__(
        self,
        *,
        target_host: str,
        target_port: int,
        network_interface: str,
        domain_id: int,
        max_rate_hz: float,
    ) -> None:
        if not target_host or not 1 <= target_port <= 65535:
            raise ValueError("target host and port must be valid")
        if not network_interface:
            raise ValueError("network interface must be non-empty")
        if not math.isfinite(max_rate_hz) or max_rate_hz <= 0.0:
            raise ValueError("max rate must be finite and positive")
        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        self.target = (target_host, target_port)
        self.min_period_ns = round(1_000_000_000.0 / max_rate_hz)
        self.source_epoch = time.time_ns()
        self.sequence = 0
        self.last_send_monotonic_ns = 0
        self.started_monotonic_ns = time.monotonic_ns()
        self.stop_event = threading.Event()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sent = 0
        self.rate_limited = 0
        self.invalid_motor_count = 0
        self.invalid_imu = 0
        self.invalid_nonfinite = 0
        self.first_tick: int | None = None
        self.last_tick: int | None = None

        ChannelFactoryInitialize(domain_id, network_interface)
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)

    @staticmethod
    def _tuple(value: Any, expected_size: int) -> tuple[float, ...]:
        result = tuple(float(item) for item in value)
        if len(result) != expected_size:
            raise ValueError(f"expected {expected_size} values")
        if not all(math.isfinite(item) for item in result):
            raise ArithmeticError("values must be finite")
        return result

    def callback(self, message: Any) -> None:
        now_monotonic_ns = time.monotonic_ns()
        if now_monotonic_ns - self.last_send_monotonic_ns < self.min_period_ns:
            self.rate_limited += 1
            return
        motor_state = getattr(message, "motor_state", ())
        if len(motor_state) < 29:
            self.invalid_motor_count += 1
            return
        imu_state = getattr(message, "imu_state", None)
        if imu_state is None:
            self.invalid_imu += 1
            return
        try:
            position = tuple(float(motor.q) for motor in motor_state[:29])
            velocity = tuple(float(motor.dq) for motor in motor_state[:29])
            quaternion = self._tuple(imu_state.quaternion, 4)
            gyroscope = self._tuple(imu_state.gyroscope, 3)
            accelerometer = self._tuple(imu_state.accelerometer, 3)
        except (AttributeError, TypeError, ValueError):
            self.invalid_imu += 1
            return
        if not all(math.isfinite(value) for value in (*position, *velocity)):
            self.invalid_nonfinite += 1
            return

        source_tick = int(message.tick)
        packet = DynamicCapturePacketV1(
            source_epoch=self.source_epoch,
            sequence=self.sequence,
            robot_stamp_ns=time.time_ns(),
            source_tick=source_tick,
            health_flags=REQUIRED_DYNAMIC_CAPTURE_HEALTH,
            joint_position=position,
            joint_velocity=velocity,
            imu_quaternion_wxyz=quaternion,
            imu_gyroscope=gyroscope,
            imu_accelerometer=accelerometer,
            mapping_digest=canonical_joint_mapping_digest(),
        )
        self.socket.sendto(serialize_dynamic_capture_packet(packet), self.target)
        if self.first_tick is None:
            self.first_tick = source_tick
        self.last_tick = source_tick
        self.sequence += 1
        self.sent += 1
        self.last_send_monotonic_ns = now_monotonic_ns

    def run(self, duration_sec: float) -> None:
        self.subscriber.Init(self.callback, 10)
        deadline = None if duration_sec <= 0.0 else time.monotonic() + duration_sec
        while not self.stop_event.wait(0.1):
            if deadline is not None and time.monotonic() >= deadline:
                break

    def close(self) -> None:
        self.stop_event.set()
        self.subscriber.Close()
        self.socket.close()

    def summary(self) -> dict[str, object]:
        elapsed_sec = (time.monotonic_ns() - self.started_monotonic_ns) / 1e9
        return {
            "schema": "g1_dynamic_capture_relay_summary_v1",
            "source_epoch": self.source_epoch,
            "target_host": self.target[0],
            "target_port": self.target[1],
            "sent": self.sent,
            "rate_limited": self.rate_limited,
            "invalid_motor_count": self.invalid_motor_count,
            "invalid_imu": self.invalid_imu,
            "invalid_nonfinite": self.invalid_nonfinite,
            "first_tick": self.first_tick,
            "last_tick": self.last_tick,
            "elapsed_sec": elapsed_sec,
            "send_rate_hz": self.sent / elapsed_sec if elapsed_sec > 0.0 else 0.0,
            "mapping_digest_sha256": canonical_joint_mapping_digest().hex(),
            "actuation_topics_created": 0,
            "command_capability": "structurally_unavailable",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--target-port", type=int, required=True)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--max-rate-hz", type=float, default=2_000.0)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    relay = G1DynamicCaptureRelay(
        target_host=args.target_host,
        target_port=args.target_port,
        network_interface=args.network_interface,
        domain_id=args.domain_id,
        max_rate_hz=args.max_rate_hz,
    )

    def stop(_signum: int, _frame: object) -> None:
        relay.stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        relay.run(args.duration_sec)
    finally:
        relay.close()
        print(json.dumps(relay.summary(), sort_keys=True))


if __name__ == "__main__":
    main()
