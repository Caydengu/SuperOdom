#!/usr/bin/env python3
"""Wait for one fresh, fully healthy HSROOT02 packet without joining ROS DDS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import zmq

from g1_root_state_bridge.protocol import (
    REQUIRED_ROOT_FUSION_FLAGS,
    deserialize_root_state_v2,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5575")
    parser.add_argument("--timeout-sec", type=float, default=12.0)
    parser.add_argument("--maximum-age-ms", type=float, default=250.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.timeout_sec <= 0.0 or args.maximum_age_ms <= 0.0:
        parser.error("timeouts and maximum age must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(args.endpoint)
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    deadline_ns = time.monotonic_ns() + round(args.timeout_sec * 1e9)
    rejected = 0
    try:
        while time.monotonic_ns() < deadline_ns:
            remaining_ms = max(1, (deadline_ns - time.monotonic_ns()) // 1_000_000)
            if not dict(poller.poll(int(remaining_ms))).get(socket, 0) & zmq.POLLIN:
                continue
            packet = deserialize_root_state_v2(socket.recv())
            required = (packet.health_flags & REQUIRED_ROOT_FUSION_FLAGS) == REQUIRED_ROOT_FUSION_FLAGS
            age_ms = (time.time_ns() - packet.publish_time_ns) * 1e-6
            if not required or not 0.0 <= age_ms <= args.maximum_age_ms:
                rejected += 1
                continue
            report = {
                "schema": "g1_kiss_root_state_readiness_v1",
                "status": "ready",
                "endpoint": args.endpoint,
                "sequence": packet.sequence,
                "source_epoch": packet.source_epoch,
                "estimate_time_ns": packet.estimate_time_ns,
                "publish_time_ns": packet.publish_time_ns,
                "pose_age_at_probe_ms": age_ms,
                "health_flags": int(packet.health_flags),
                "calibration_digest": packet.calibration_digest.hex(),
                "rejected_packets": rejected,
                "command_capability": "structurally_unavailable",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(report, sort_keys=True))
            return 0
    finally:
        socket.close()
        context.term()
    raise TimeoutError(
        f"no fresh, fully healthy HSROOT02 packet from {args.endpoint} "
        f"within {args.timeout_sec:.1f}s"
    )


if __name__ == "__main__":
    raise SystemExit(main())
