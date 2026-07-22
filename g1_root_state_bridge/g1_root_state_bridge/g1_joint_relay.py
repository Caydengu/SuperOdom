"""Read-only Unitree G1 joint sidecar.

This process runs in the robot's Python environment that contains Unitree's
generated CycloneDDS IDL. It never creates a command publisher. Each accepted
LowState callback is stamped on the G1 and sent as a fixed, CRC-protected UDP
datagram to the offboard pelvis bridge.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import socket
import threading
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

from g1_root_state_bridge.joint_transport import (
    REQUIRED_JOINT_HEALTH_FLAGS,
    JointPacketV1,
    canonical_joint_mapping_digest,
    serialize_joint_packet,
)


class G1JointRelay:
    def __init__(
        self,
        *,
        target_host: str,
        target_port: int,
        network_interface: str,
        domain_id: int,
        max_rate_hz: float,
    ):
        self.target = (target_host, target_port)
        self.min_period_ns = int(1e9 / max_rate_hz)
        self.source_epoch = time.time_ns()
        self.sequence = 0
        self.last_send_monotonic_ns = 0
        self.sent = 0
        self.rate_limited = 0
        self.invalid = 0
        self.first_tick: int | None = None
        self.last_tick: int | None = None
        self.started_monotonic_ns = time.monotonic_ns()
        self.stop_event = threading.Event()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        ChannelFactoryInitialize(domain_id, network_interface)
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)

    def callback(self, message: LowState_) -> None:
        now_monotonic_ns = time.monotonic_ns()
        if now_monotonic_ns - self.last_send_monotonic_ns < self.min_period_ns:
            self.rate_limited += 1
            return
        if len(message.motor_state) < 29:
            self.invalid += 1
            return

        position = tuple(float(motor.q) for motor in message.motor_state[:29])
        velocity = tuple(float(motor.dq) for motor in message.motor_state[:29])
        if not all(math.isfinite(value) for value in (*position, *velocity)):
            self.invalid += 1
            return

        stamp_ns = time.time_ns()
        source_tick = int(message.tick)
        packet = JointPacketV1(
            source_epoch=self.source_epoch,
            sequence=self.sequence,
            stamp_ns=stamp_ns,
            source_tick=source_tick,
            health_flags=REQUIRED_JOINT_HEALTH_FLAGS,
            position=position,
            velocity=velocity,
            mapping_digest=canonical_joint_mapping_digest(),
        )
        self.socket.sendto(serialize_joint_packet(packet), self.target)
        if self.first_tick is None:
            self.first_tick = source_tick
        self.last_tick = source_tick
        self.sequence += 1
        self.sent += 1
        self.last_send_monotonic_ns = now_monotonic_ns

    def run(self, duration_sec: float) -> None:
        self.subscriber.Init(self.callback, 10)
        deadline = None if duration_sec <= 0.0 else time.monotonic() + duration_sec
        while not self.stop_event.wait(timeout=0.1):
            if deadline is not None and time.monotonic() >= deadline:
                break

    def close(self) -> None:
        self.stop_event.set()
        self.subscriber.Close()
        self.socket.close()

    def summary(self) -> dict[str, object]:
        elapsed_sec = (time.monotonic_ns() - self.started_monotonic_ns) / 1e9
        return {
            "schema": "g1_joint_relay_summary_v1",
            "source_epoch": self.source_epoch,
            "target_host": self.target[0],
            "target_port": self.target[1],
            "sent": self.sent,
            "rate_limited": self.rate_limited,
            "invalid": self.invalid,
            "first_tick": self.first_tick,
            "last_tick": self.last_tick,
            "elapsed_sec": elapsed_sec,
            "send_rate_hz": self.sent / elapsed_sec if elapsed_sec > 0.0 else 0.0,
            "mapping_digest_sha256": canonical_joint_mapping_digest().hex(),
            "actuation_topics_created": 0,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--target-port", type=int, default=5576)
    parser.add_argument("--network-interface", default="eth0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--max-rate-hz", type=float, default=2000.0)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.target_port <= 65535:
        raise SystemExit("target port must be from 1 through 65535")
    if args.max_rate_hz <= 0.0:
        raise SystemExit("max rate must be positive")
    relay = G1JointRelay(
        target_host=args.target_host,
        target_port=args.target_port,
        network_interface=args.network_interface,
        domain_id=args.domain_id,
        max_rate_hz=args.max_rate_hz,
    )

    def stop(_signum, _frame) -> None:
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
