from __future__ import annotations

from dataclasses import replace
import math

import pytest

from g1_root_state_bridge.protocol import (
    REQUIRED_HEALTH_FLAGS,
    ROOT_STATE_V2_MAGIC,
    ROOT_STATE_V2_NUM_BYTES,
    RootStateHealth,
    RootStatePacketV2,
    RootStateProtocolError,
    deserialize_root_state_v2,
    serialize_root_state_v2,
)


def packet() -> RootStatePacketV2:
    return RootStatePacketV2(
        sequence=42,
        source_epoch=7,
        estimate_time_ns=1_000_000_000,
        publish_time_ns=1_005_000_000,
        correction_time_ns=950_000_000,
        joint_time_ns=999_000_000,
        joint_sync_gap_ns=-1_000_000,
        health_flags=REQUIRED_HEALTH_FLAGS,
        position=(1.25, -0.5, 0.75),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.125, -0.25, 0.5),
        angular_velocity=(0.0, 0.1, -0.2),
        covariance_diagonal=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
        calibration_digest=bytes(range(32)),
    )


def test_v2_round_trip_has_fixed_little_endian_contract() -> None:
    encoded = serialize_root_state_v2(packet())
    decoded = deserialize_root_state_v2(encoded)

    assert ROOT_STATE_V2_MAGIC == b"HSROOT02"
    assert len(encoded) == ROOT_STATE_V2_NUM_BYTES == 176
    assert encoded[:8] == ROOT_STATE_V2_MAGIC
    assert decoded.sequence == 42
    assert decoded.source_epoch == 7
    assert decoded.estimate_time_ns == 1_000_000_000
    assert decoded.publish_time_ns == 1_005_000_000
    assert decoded.correction_time_ns == 950_000_000
    assert decoded.joint_time_ns == 999_000_000
    assert decoded.joint_sync_gap_ns == -1_000_000
    assert decoded.health_flags == REQUIRED_HEALTH_FLAGS
    assert decoded.position == (1.25, -0.5, 0.75)
    assert decoded.quaternion_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert decoded.linear_velocity == (0.125, -0.25, 0.5)
    assert decoded.calibration_digest == bytes(range(32))


def test_required_health_bits_are_independent_and_complete() -> None:
    expected = (
        RootStateHealth.ESTIMATOR_HEALTHY
        | RootStateHealth.FINITE_POSE
        | RootStateHealth.JOINT_SYNC_VALID
        | RootStateHealth.CORRECTION_FRESH
        | RootStateHealth.CALIBRATION_VALID
        | RootStateHealth.CLOCK_VALID
    )
    assert REQUIRED_HEALTH_FLAGS == expected


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"sequence": -1}, "sequence"),
        ({"publish_time_ns": 999_999_999}, "publish"),
        ({"correction_time_ns": 1_000_000_001}, "correction"),
        ({"joint_sync_gap_ns": 0}, "joint_sync_gap"),
        ({"position": (math.nan, 0.0, 0.0)}, "finite"),
        ({"quaternion_wxyz": (2.0, 0.0, 0.0, 0.0)}, "quaternion"),
        ({"covariance_diagonal": (0.01, 0.02, -0.03, 0.04, 0.05, 0.06)}, "covariance"),
        ({"calibration_digest": b"short"}, "digest"),
    ],
)
def test_v2_serializer_rejects_invalid_semantics(mutation: dict[str, object], match: str) -> None:
    with pytest.raises(RootStateProtocolError, match=match):
        serialize_root_state_v2(replace(packet(), **mutation))


def test_v2_decoder_rejects_wrong_magic_and_length() -> None:
    encoded = serialize_root_state_v2(packet())
    with pytest.raises(RootStateProtocolError, match="length"):
        deserialize_root_state_v2(encoded[:-1])
    with pytest.raises(RootStateProtocolError, match="magic"):
        deserialize_root_state_v2(b"BADMAGIC" + encoded[8:])
