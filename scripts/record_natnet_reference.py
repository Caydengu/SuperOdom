#!/usr/bin/env python3
"""Record one Motive rigid body with native coordinates and host receipt clocks."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any


SCHEMA = "g1_optitrack_raw_v1"


def parse_rigid_body_id(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("rigid-body ID must be nonnegative or 'auto'")
    return parsed


def frame_record(
    data: dict[str, Any],
    *,
    target_id: int,
    target_name: str,
    receipt_realtime_ns: int,
    receipt_monotonic_ns: int,
) -> dict[str, Any] | None:
    rigid_body_data = data.get("rigid_body_data")
    bodies = getattr(rigid_body_data, "rigid_body_list", ())
    body = next((candidate for candidate in bodies if int(candidate.id_num) == target_id), None)
    if body is None:
        return None
    return {
        "schema": SCHEMA,
        "record_type": "frame",
        "frame_number": int(data["frame_number"]),
        "motive_software_time_s": float(data["timestamp"]),
        "motive_is_recording": bool(data.get("is_recording", False)),
        "rigid_body_id": target_id,
        "rigid_body_name": target_name,
        "tracking_valid": bool(body.tracking_valid),
        "position_xyz_m_motive_native": [float(value) for value in body.pos],
        "quaternion_xyzw_motive_native": [float(value) for value in body.rot],
        "mean_marker_error_m": float(getattr(body, "error", 0.0)),
        "receipt_realtime_ns": receipt_realtime_ns,
        "receipt_monotonic_ns": receipt_monotonic_ns,
    }


class Recorder:
    def __init__(self, output: Path, target_id: int | None, target_name: str) -> None:
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.stream = output.open("x", encoding="utf-8", buffering=1)
        self.target_id = target_id
        self.target_name = target_name
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.discovered_ids: Counter[int] = Counter()
        self.frames_seen = 0
        self.frames_written = 0
        self.valid_frames = 0
        self.missing_target_frames = 0
        self.duplicate_or_reordered = 0
        self.first_frame: int | None = None
        self.last_frame: int | None = None
        self.started_realtime_ns = time.time_ns()
        self.started_monotonic_ns = time.monotonic_ns()

    def write(self, record: dict[str, Any]) -> None:
        self.stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def metadata(self, args: argparse.Namespace) -> None:
        self.write(
            {
                "schema": SCHEMA,
                "record_type": "metadata",
                "server_address": args.server_address,
                "client_address": args.client_address,
                "connection": args.connection,
                "multicast_address": args.multicast_address,
                "requested_rigid_body_id": args.rigid_body_id,
                "rigid_body_name": args.rigid_body_name,
                "coordinate_contract": "native Motive coordinates; no online tare or frame conversion",
                "started_realtime_ns": self.started_realtime_ns,
                "started_monotonic_ns": self.started_monotonic_ns,
            }
        )

    def callback(self, data: dict[str, Any]) -> None:
        receipt_monotonic_ns = time.monotonic_ns()
        receipt_realtime_ns = time.time_ns()
        rigid_body_data = data.get("rigid_body_data")
        bodies = tuple(getattr(rigid_body_data, "rigid_body_list", ()))
        with self.lock:
            self.frames_seen += 1
            for body in bodies:
                self.discovered_ids[int(body.id_num)] += 1
            if self.target_id is None:
                ids = sorted({int(body.id_num) for body in bodies})
                if len(ids) == 1:
                    self.target_id = ids[0]
            if self.target_id is None:
                self.missing_target_frames += 1
                return
            record = frame_record(
                data,
                target_id=self.target_id,
                target_name=self.target_name,
                receipt_realtime_ns=receipt_realtime_ns,
                receipt_monotonic_ns=receipt_monotonic_ns,
            )
            if record is None:
                self.missing_target_frames += 1
                return
            frame_number = int(record["frame_number"])
            if self.last_frame is not None and frame_number <= self.last_frame:
                self.duplicate_or_reordered += 1
                return
            if self.first_frame is None:
                self.first_frame = frame_number
            self.last_frame = frame_number
            self.frames_written += 1
            self.valid_frames += int(record["tracking_valid"])
            self.write(record)

    def close(self) -> None:
        with self.lock:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()

    def summary(self, *, connected: bool, elapsed_sec: float) -> dict[str, Any]:
        coverage = self.valid_frames / self.frames_written if self.frames_written else 0.0
        return {
            "schema": "g1_optitrack_raw_summary_v1",
            "output": str(self.output.resolve()),
            "connected": connected,
            "resolved_rigid_body_id": self.target_id,
            "rigid_body_name": self.target_name,
            "discovered_rigid_body_frame_counts": dict(sorted(self.discovered_ids.items())),
            "frames_seen": self.frames_seen,
            "frames_written": self.frames_written,
            "tracking_valid_frames": self.valid_frames,
            "tracking_coverage": coverage,
            "missing_target_frames": self.missing_target_frames,
            "duplicate_or_reordered_frames": self.duplicate_or_reordered,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "elapsed_sec": elapsed_sec,
            "write_rate_hz": self.frames_written / elapsed_sec if elapsed_sec > 0.0 else 0.0,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-root", type=Path, required=True, help="Directory containing mocap_utils/")
    parser.add_argument("--server-address", required=True)
    parser.add_argument("--client-address", required=True)
    parser.add_argument("--connection", choices=("multicast", "unicast"), default="multicast")
    parser.add_argument("--multicast-address", default="239.255.42.99")
    parser.add_argument("--rigid-body-id", type=parse_rigid_body_id, default=None)
    parser.add_argument("--rigid-body-name", default="G1_PELVIS")
    parser.add_argument("--duration-sec", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--minimum-tracking-coverage", type=float, default=0.95)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.duration_sec <= 0.0:
        raise SystemExit("--duration-sec must be positive")
    if not 0.0 <= args.minimum_tracking_coverage <= 1.0:
        raise SystemExit("--minimum-tracking-coverage must be in [0, 1]")
    if args.summary.exists():
        raise FileExistsError(args.summary)

    sdk_root = args.sdk_root.resolve()
    package_root = sdk_root / "mocap_utils"
    if not package_root.is_dir():
        raise FileNotFoundError(f"NatNet SDK package is missing: {package_root}")
    # NaturalPoint's Python SDK imports its sibling modules as top-level names,
    # even when natnet_client itself is imported through the mocap_utils package.
    sys.path.insert(0, str(package_root))
    sys.path.insert(0, str(sdk_root))
    natnet_module = importlib.import_module("mocap_utils.natnet_client")
    client = natnet_module.NatNetClient()
    client.set_client_address(args.client_address)
    client.set_server_address(args.server_address)
    client.set_use_multicast(args.connection == "multicast")
    if hasattr(client, "multicast_address"):
        client.multicast_address = args.multicast_address
    if hasattr(client, "set_print_level"):
        client.set_print_level(0)

    recorder = Recorder(args.output, args.rigid_body_id, args.rigid_body_name)
    recorder.metadata(args)
    client.new_frame_listener = recorder.callback

    def stop(_signum: int, _frame: object) -> None:
        recorder.stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    if not client.run():
        recorder.close()
        raise SystemExit("NatNet client failed to start")
    # The bundled SDK leaves its multicast data socket blocking forever during shutdown.
    client.data_socket.settimeout(0.1)
    client.command_socket.settimeout(0.1)
    try:
        deadline = started + args.duration_sec
        while not recorder.stop_event.wait(0.05) and time.monotonic() < deadline:
            pass
    finally:
        connected = bool(client.connected())
        client.stop_threads = True
        for stream_socket in (client.command_socket, client.data_socket):
            stream_socket.close()
        elapsed_sec = time.monotonic() - started
        recorder.close()

    summary = recorder.summary(connected=connected, elapsed_sec=elapsed_sec)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(summary, sort_keys=True), flush=True)
    exit_code = 0
    if not connected or summary["frames_written"] == 0:
        exit_code = 2
    elif summary["tracking_coverage"] < args.minimum_tracking_coverage:
        exit_code = 3
    # NatNet's non-daemon receive threads can remain in recvfrom after close on Linux.
    os._exit(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
