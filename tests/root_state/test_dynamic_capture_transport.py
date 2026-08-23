from __future__ import annotations

from dataclasses import replace
import io
from pathlib import Path

import pytest

from g1_root_state_bridge.dynamic_capture_io import (
    RecordedDynamicPacket,
    iter_recorded_datagrams,
    write_recorded_datagram,
)
from g1_root_state_bridge.dynamic_capture_transport import (
    DYNAMIC_CAPTURE_PACKET_NUM_BYTES,
    REQUIRED_DYNAMIC_CAPTURE_HEALTH,
    DynamicCaptureHealth,
    DynamicCapturePacketError,
    DynamicCapturePacketV1,
    deserialize_dynamic_capture_packet,
    serialize_dynamic_capture_packet,
)
from g1_root_state_bridge.joint_transport import canonical_joint_mapping_digest


def packet(**mutations: object) -> DynamicCapturePacketV1:
    base = DynamicCapturePacketV1(
        source_epoch=10,
        sequence=11,
        robot_stamp_ns=12,
        source_tick=13,
        health_flags=REQUIRED_DYNAMIC_CAPTURE_HEALTH,
        joint_position=tuple(float(index) for index in range(29)),
        joint_velocity=tuple(float(index + 100) for index in range(29)),
        imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        imu_gyroscope=(0.125, 0.25, 0.5),
        imu_accelerometer=(0.0, 0.0, 9.8125),
        mapping_digest=canonical_joint_mapping_digest(),
    )
    return replace(base, **mutations)


def test_dynamic_capture_packet_has_fixed_exact_round_trip() -> None:
    original = packet()

    encoded = serialize_dynamic_capture_packet(original)
    decoded = deserialize_dynamic_capture_packet(encoded)

    assert len(encoded) == DYNAMIC_CAPTURE_PACKET_NUM_BYTES
    assert decoded == original
    assert decoded.strictly_valid


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"joint_position": tuple(0.0 for _ in range(28))}, "29 values"),
        ({"joint_velocity": tuple(0.0 for _ in range(30))}, "29 values"),
        ({"imu_quaternion_wxyz": (1.0, 0.0, 0.0)}, "four values"),
        ({"imu_gyroscope": (0.0, 0.0)}, "three values"),
        ({"imu_accelerometer": (0.0, 0.0, 0.0, 0.0)}, "three values"),
        ({"joint_position": (float("nan"),) + tuple(0.0 for _ in range(28))}, "finite"),
        ({"imu_gyroscope": (float("inf"), 0.0, 0.0)}, "finite"),
        ({"mapping_digest": b"short"}, "32 raw bytes"),
    ],
)
def test_dynamic_capture_packet_rejects_malformed_fields(
    mutation: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(DynamicCapturePacketError, match=message):
        serialize_dynamic_capture_packet(packet(**mutation))


def test_dynamic_capture_packet_rejects_crc_magic_and_truncation() -> None:
    encoded = bytearray(serialize_dynamic_capture_packet(packet()))

    corrupted = encoded.copy()
    corrupted[-5] ^= 0x80
    with pytest.raises(DynamicCapturePacketError, match="CRC"):
        deserialize_dynamic_capture_packet(bytes(corrupted))

    wrong_magic = encoded.copy()
    wrong_magic[0:8] = b"BADMAGIC"
    with pytest.raises(DynamicCapturePacketError, match="magic"):
        deserialize_dynamic_capture_packet(bytes(wrong_magic))

    with pytest.raises(DynamicCapturePacketError, match="length"):
        deserialize_dynamic_capture_packet(bytes(encoded[:-1]))


def test_missing_health_bit_is_well_formed_but_not_strictly_valid() -> None:
    reduced = REQUIRED_DYNAMIC_CAPTURE_HEALTH & ~DynamicCaptureHealth.IMU_VALID
    decoded = deserialize_dynamic_capture_packet(
        serialize_dynamic_capture_packet(packet(health_flags=reduced))
    )

    assert not decoded.strictly_valid


def test_recorded_datagrams_preserve_both_oslo_clocks_and_source_bytes(
    tmp_path: Path,
) -> None:
    source_packets = [
        serialize_dynamic_capture_packet(packet(sequence=sequence))
        for sequence in (20, 21, 22)
    ]
    output = tmp_path / "lowstate.hsdyn"
    with output.open("xb") as stream:
        for index, payload in enumerate(source_packets):
            write_recorded_datagram(
                stream,
                payload,
                receipt_realtime_ns=1_000 + index,
                receipt_monotonic_ns=2_000 + index,
            )

    records = list(iter_recorded_datagrams(output))

    assert records == [
        RecordedDynamicPacket(
            packet=deserialize_dynamic_capture_packet(payload),
            source_bytes=payload,
            receipt_realtime_ns=1_000 + index,
            receipt_monotonic_ns=2_000 + index,
        )
        for index, payload in enumerate(source_packets)
    ]


def test_recorded_datagrams_reject_truncation_and_nonmonotonic_receipt(
    tmp_path: Path,
) -> None:
    payload = serialize_dynamic_capture_packet(packet())
    complete = io.BytesIO()
    write_recorded_datagram(
        complete,
        payload,
        receipt_realtime_ns=1_000,
        receipt_monotonic_ns=2_000,
    )
    truncated = tmp_path / "truncated.hsdyn"
    truncated.write_bytes(complete.getvalue()[:-1])
    with pytest.raises(DynamicCapturePacketError, match="truncated"):
        list(iter_recorded_datagrams(truncated))

    nonmonotonic = tmp_path / "nonmonotonic.hsdyn"
    with nonmonotonic.open("xb") as stream:
        write_recorded_datagram(
            stream,
            serialize_dynamic_capture_packet(packet(sequence=20)),
            receipt_realtime_ns=1_000,
            receipt_monotonic_ns=2_000,
        )
        write_recorded_datagram(
            stream,
            serialize_dynamic_capture_packet(packet(sequence=21)),
            receipt_realtime_ns=1_001,
            receipt_monotonic_ns=1_999,
        )
    with pytest.raises(DynamicCapturePacketError, match="receipt monotonic"):
        list(iter_recorded_datagrams(nonmonotonic))


def test_recorded_datagrams_reject_duplicate_source_sequence(tmp_path: Path) -> None:
    output = tmp_path / "duplicate.hsdyn"
    with output.open("xb") as stream:
        for receipt in (2_000, 2_001):
            write_recorded_datagram(
                stream,
                serialize_dynamic_capture_packet(packet(sequence=20)),
                receipt_realtime_ns=1_000,
                receipt_monotonic_ns=receipt,
            )

    with pytest.raises(DynamicCapturePacketError, match="source sequence"):
        list(iter_recorded_datagrams(output))
