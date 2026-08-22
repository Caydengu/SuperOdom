from __future__ import annotations

from dataclasses import replace

import pytest

from g1_root_state_bridge.root_imu_transport import (
    REQUIRED_ROOT_IMU_HEALTH_FLAGS,
    RootImuPacketDecoder,
    RootImuPacketError,
    RootImuPacketV1,
    deserialize_root_imu_packet,
    serialize_root_imu_packet,
)


def packet(**changes: object) -> RootImuPacketV1:
    base = RootImuPacketV1(
        source_epoch=7,
        sequence=4,
        stamp_ns=1_000_000_000,
        source_tick=42,
        health_flags=REQUIRED_ROOT_IMU_HEALTH_FLAGS,
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        angular_velocity=(0.1, -0.2, 0.3),
        linear_acceleration=(0.0, 0.0, 9.81),
    )
    return replace(base, **changes)


def test_root_imu_packet_round_trip_and_crc() -> None:
    encoded = serialize_root_imu_packet(packet())
    decoded = deserialize_root_imu_packet(encoded)
    assert decoded.source_epoch == packet().source_epoch
    assert decoded.sequence == packet().sequence
    assert decoded.stamp_ns == packet().stamp_ns
    assert decoded.quaternion_wxyz == pytest.approx(packet().quaternion_wxyz)
    assert decoded.angular_velocity == pytest.approx(packet().angular_velocity)
    assert decoded.linear_acceleration == pytest.approx(packet().linear_acceleration)
    damaged = bytearray(encoded)
    damaged[-5] ^= 1
    with pytest.raises(RootImuPacketError, match="CRC"):
        deserialize_root_imu_packet(bytes(damaged))


def test_root_imu_decoder_rejects_stale_and_nonmonotonic_samples() -> None:
    decoder = RootImuPacketDecoder(max_age_ms=10.0)
    sample = decoder.decode(
        serialize_root_imu_packet(packet()), receipt_ns=1_005_000_000
    )
    assert sample.source_epoch == 7
    with pytest.raises(RootImuPacketError, match="sequence"):
        decoder.decode(
            serialize_root_imu_packet(packet()), receipt_ns=1_006_000_000
        )
    with pytest.raises(RootImuPacketError, match="stale"):
        RootImuPacketDecoder(max_age_ms=1.0).decode(
            serialize_root_imu_packet(packet()), receipt_ns=1_005_000_000
        )
