"""Passive UDP receiver for CRC-protected G1 DynamicCapturePacketV1 samples."""

from __future__ import annotations

import socket
import threading
import time
from typing import Callable

from g1_root_state_bridge.dynamic_capture_transport import (
    DYNAMIC_CAPTURE_PACKET_NUM_BYTES,
    DynamicCapturePacketV1,
    deserialize_dynamic_capture_packet,
)


class DynamicLowStateReceiver:
    """Receive only; this class has no Unitree command channel or publisher."""

    def __init__(
        self,
        *,
        bind_host: str,
        bind_port: int,
        callback: Callable[[DynamicCapturePacketV1, int], None],
    ) -> None:
        if not 1 <= bind_port <= 65535:
            raise ValueError("bind port must be in [1,65535]")
        self.callback = callback
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        self.socket.bind((bind_host, bind_port))
        self.socket.settimeout(0.2)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="g1-lowstate-receiver", daemon=True)
        self.accepted = 0
        self.invalid = 0

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                payload, _source = self.socket.recvfrom(DYNAMIC_CAPTURE_PACKET_NUM_BYTES + 1)
            except socket.timeout:
                continue
            receipt_time_ns = time.time_ns()
            try:
                packet = deserialize_dynamic_capture_packet(payload)
            except ValueError:
                self.invalid += 1
                continue
            self.accepted += 1
            self.callback(packet, receipt_time_ns)

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self.socket.close()
