"""Lossless fixed-record storage for G1 dynamic-capture datagrams."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import BinaryIO, Iterator

from g1_root_state_bridge.dynamic_capture_transport import (
    DYNAMIC_CAPTURE_PACKET_NUM_BYTES,
    DynamicCapturePacketError,
    DynamicCapturePacketV1,
    deserialize_dynamic_capture_packet,
)


RECORDED_DATAGRAM_MAGIC = b"HSDREC01"
RECORDED_DATAGRAM_HEADER = struct.Struct("<8sIqqI")


@dataclass(frozen=True)
class RecordedDynamicPacket:
    packet: DynamicCapturePacketV1
    source_bytes: bytes
    receipt_realtime_ns: int
    receipt_monotonic_ns: int


def write_recorded_datagram(
    stream: BinaryIO,
    packet_bytes: bytes,
    *,
    receipt_realtime_ns: int,
    receipt_monotonic_ns: int,
) -> None:
    """Append one source datagram and both Oslo receive clocks."""

    deserialize_dynamic_capture_packet(packet_bytes)
    if receipt_realtime_ns < 0 or receipt_monotonic_ns < 0:
        raise DynamicCapturePacketError(
            "recorded receipt timestamps must be non-negative"
        )
    header = RECORDED_DATAGRAM_HEADER.pack(
        RECORDED_DATAGRAM_MAGIC,
        RECORDED_DATAGRAM_HEADER.size + len(packet_bytes),
        receipt_realtime_ns,
        receipt_monotonic_ns,
        len(packet_bytes),
    )
    stream.write(header)
    stream.write(packet_bytes)


def iter_recorded_datagrams(path: str | Path) -> Iterator[RecordedDynamicPacket]:
    """Yield authenticated records while enforcing source and receipt order."""

    source = Path(path)
    last_receipt_monotonic_ns: int | None = None
    last_source_epoch: int | None = None
    last_sequence: int | None = None
    with source.open("rb") as stream:
        while True:
            header = stream.read(RECORDED_DATAGRAM_HEADER.size)
            if not header:
                return
            if len(header) != RECORDED_DATAGRAM_HEADER.size:
                raise DynamicCapturePacketError(
                    "recorded datagram header is truncated"
                )
            (
                magic,
                record_size,
                receipt_realtime_ns,
                receipt_monotonic_ns,
                packet_size,
            ) = RECORDED_DATAGRAM_HEADER.unpack(header)
            if magic != RECORDED_DATAGRAM_MAGIC:
                raise DynamicCapturePacketError(
                    "recorded datagram magic mismatch"
                )
            if packet_size != DYNAMIC_CAPTURE_PACKET_NUM_BYTES:
                raise DynamicCapturePacketError(
                    "recorded datagram source packet length mismatch"
                )
            if record_size != RECORDED_DATAGRAM_HEADER.size + packet_size:
                raise DynamicCapturePacketError(
                    "recorded datagram record size mismatch"
                )
            payload = stream.read(packet_size)
            if len(payload) != packet_size:
                raise DynamicCapturePacketError(
                    "recorded datagram source packet is truncated"
                )
            if receipt_realtime_ns < 0 or receipt_monotonic_ns < 0:
                raise DynamicCapturePacketError(
                    "recorded receipt timestamps must be non-negative"
                )
            if (
                last_receipt_monotonic_ns is not None
                and receipt_monotonic_ns <= last_receipt_monotonic_ns
            ):
                raise DynamicCapturePacketError(
                    "recorded receipt monotonic timestamp is not increasing"
                )
            packet = deserialize_dynamic_capture_packet(payload)
            if last_source_epoch is not None:
                if packet.source_epoch < last_source_epoch:
                    raise DynamicCapturePacketError(
                        "source epoch is not increasing"
                    )
                if (
                    packet.source_epoch == last_source_epoch
                    and last_sequence is not None
                    and packet.sequence <= last_sequence
                ):
                    raise DynamicCapturePacketError(
                        "source sequence is not increasing"
                    )
            yield RecordedDynamicPacket(
                packet=packet,
                source_bytes=payload,
                receipt_realtime_ns=receipt_realtime_ns,
                receipt_monotonic_ns=receipt_monotonic_ns,
            )
            last_receipt_monotonic_ns = receipt_monotonic_ns
            last_source_epoch = packet.source_epoch
            last_sequence = packet.sequence
