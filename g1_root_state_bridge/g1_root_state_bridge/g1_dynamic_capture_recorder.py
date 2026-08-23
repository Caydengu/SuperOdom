"""Lossless Oslo-side recorder for typed G1 dynamic-capture UDP packets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import tempfile
import threading
import time

from g1_root_state_bridge.dynamic_capture_io import write_recorded_datagram
from g1_root_state_bridge.dynamic_capture_transport import (
    DynamicCapturePacketError,
    deserialize_dynamic_capture_packet,
)


class G1DynamicCaptureRecorder:
    """Persist valid increasing source packets with both Oslo receive clocks."""

    def __init__(
        self,
        *,
        bind_host: str,
        bind_port: int,
        output_path: str | Path,
        summary_path: str | Path,
        socket_timeout_s: float = 0.05,
        fsync_period_s: float = 0.5,
    ) -> None:
        if not bind_host or not 0 <= bind_port <= 65535:
            raise ValueError("bind host and port must be valid")
        if socket_timeout_s <= 0.0 or fsync_period_s <= 0.0:
            raise ValueError("timeout and fsync period must be positive")
        self.output_path = Path(output_path)
        self.summary_path = Path(summary_path)
        if self.output_path.exists():
            raise FileExistsError(self.output_path)
        if self.summary_path.exists():
            raise FileExistsError(self.summary_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.output_path.open("xb")
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((bind_host, bind_port))
        self._socket.settimeout(socket_timeout_s)
        self.bound_port = int(self._socket.getsockname()[1])
        self.fsync_period_s = fsync_period_s
        self.stop_event = threading.Event()
        self.started_monotonic_ns = time.monotonic_ns()
        self.accepted = 0
        self.invalid = 0
        self.sequence_gaps = 0
        self.duplicates_or_reordered = 0
        self.source_epochs: set[int] = set()
        self.first_sequence: int | None = None
        self.last_sequence: int | None = None
        self._last_source_epoch: int | None = None
        self._last_sequence: int | None = None

    def _accept_source_order(self, source_epoch: int, sequence: int) -> bool:
        if self._last_source_epoch is None:
            return True
        if source_epoch < self._last_source_epoch:
            self.duplicates_or_reordered += 1
            return False
        if source_epoch > self._last_source_epoch:
            return True
        assert self._last_sequence is not None
        if sequence <= self._last_sequence:
            self.duplicates_or_reordered += 1
            return False
        if sequence > self._last_sequence + 1:
            self.sequence_gaps += sequence - self._last_sequence - 1
        return True

    def run(self, duration_sec: float) -> None:
        if duration_sec <= 0.0:
            raise ValueError("duration must be positive")
        deadline = time.monotonic() + duration_sec
        next_fsync = time.monotonic() + self.fsync_period_s
        try:
            while not self.stop_event.is_set() and time.monotonic() < deadline:
                try:
                    payload, _address = self._socket.recvfrom(65_535)
                except TimeoutError:
                    continue
                receipt_monotonic_ns = time.monotonic_ns()
                receipt_realtime_ns = time.time_ns()
                try:
                    packet = deserialize_dynamic_capture_packet(payload)
                except DynamicCapturePacketError:
                    self.invalid += 1
                    continue
                if not packet.strictly_valid:
                    self.invalid += 1
                    continue
                if not self._accept_source_order(
                    packet.source_epoch, packet.sequence
                ):
                    continue
                write_recorded_datagram(
                    self._stream,
                    payload,
                    receipt_realtime_ns=receipt_realtime_ns,
                    receipt_monotonic_ns=receipt_monotonic_ns,
                )
                self.accepted += 1
                self.source_epochs.add(packet.source_epoch)
                if self.first_sequence is None:
                    self.first_sequence = packet.sequence
                self.last_sequence = packet.sequence
                self._last_source_epoch = packet.source_epoch
                self._last_sequence = packet.sequence
                if time.monotonic() >= next_fsync:
                    self._stream.flush()
                    os.fsync(self._stream.fileno())
                    next_fsync = time.monotonic() + self.fsync_period_s
        finally:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            self._socket.close()
            self._write_summary()

    def stop(self) -> None:
        self.stop_event.set()

    def summary(self) -> dict[str, object]:
        elapsed_sec = (time.monotonic_ns() - self.started_monotonic_ns) / 1e9
        return {
            "schema": "g1_dynamic_capture_summary_v1",
            "output_path": str(self.output_path.resolve()),
            "bind_port": self.bound_port,
            "accepted": self.accepted,
            "invalid": self.invalid,
            "sequence_gaps": self.sequence_gaps,
            "duplicates_or_reordered": self.duplicates_or_reordered,
            "source_epochs": sorted(self.source_epochs),
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "elapsed_sec": elapsed_sec,
            "accepted_rate_hz": (
                self.accepted / elapsed_sec if elapsed_sec > 0.0 else 0.0
            ),
            "actuation_topics_created": 0,
            "command_capability": "structurally_unavailable",
        }

    def _write_summary(self) -> None:
        payload = json.dumps(self.summary(), indent=2, sort_keys=True) + "\n"
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                prefix=f".{self.summary_path.name}.",
                suffix=".tmp",
                dir=self.summary_path.parent,
                delete=False,
            ) as stream:
                temporary_name = stream.name
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.summary_path)
        finally:
            if temporary_name is not None and os.path.exists(temporary_name):
                os.unlink(temporary_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind-host", default="0.0.0.0")
    parser.add_argument("--bind-port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--duration-sec", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recorder = G1DynamicCaptureRecorder(
        bind_host=args.bind_host,
        bind_port=args.bind_port,
        output_path=args.output,
        summary_path=args.summary,
    )

    def stop(_signum: int, _frame: object) -> None:
        recorder.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    recorder.run(args.duration_sec)
    print(json.dumps(recorder.summary(), sort_keys=True))


if __name__ == "__main__":
    main()
