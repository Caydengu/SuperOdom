from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import math

import pytest

from g1_root_state_bridge.joint_contract import (
    CANONICAL_G1_JOINT_NAMES,
    TimedJointSample,
)
from g1_root_state_bridge.joint_transport import canonical_joint_mapping_digest
from g1_root_state_bridge.policy_state_protocol import (
    G1_POLICY_STATE_V1_NUM_BYTES,
    G1PolicyStatePacketV1,
    PolicyStateHealth,
    PolicyStateProtocolError,
    build_policy_state_v1,
    deserialize_policy_state_v1,
    serialize_policy_state_v1,
)
from g1_root_state_bridge.protocol import (
    REQUIRED_HEALTH_FLAGS,
    RootStateHealth,
    RootStatePacketV2,
)


GOLDEN_SHA256 = "e5fa21f60ae38fc04d7dccdda87a57847f8c060b43ce8dfc3a71a882fb6fd34b"


def golden_packet() -> G1PolicyStatePacketV1:
    return G1PolicyStatePacketV1(
        sequence=0x0102030405060708,
        source_epoch=0x1112131415161718,
        estimate_time_ns=1_700_000_000_100_000_000,
        publish_time_ns=1_700_000_000_102_000_000,
        correction_time_ns=1_700_000_000_000_000_000,
        joint_time_ns=1_700_000_000_101_000_000,
        joint_sync_gap_ns=1_000_000,
        health_flags=PolicyStateHealth(0xFF),
        position=(1.25, -2.5, 0.75),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.1, -0.2, 0.3),
        angular_velocity=(-0.4, 0.5, -0.6),
        covariance_diagonal=(0.01, 0.02, 0.03, 0.04, 0.05, 0.06),
        joint_position=tuple((index - 14) / 10 for index in range(29)),
        joint_velocity=tuple((14 - index) / 100 for index in range(29)),
        calibration_digest=bytes(range(32)),
        joint_mapping_digest=bytes(range(32, 64)),
    )


def root_packet() -> RootStatePacketV2:
    return RootStatePacketV2(
        sequence=7,
        source_epoch=9,
        estimate_time_ns=2_000_000_000,
        publish_time_ns=2_002_000_000,
        correction_time_ns=1_900_000_000,
        joint_time_ns=2_001_000_000,
        joint_sync_gap_ns=1_000_000,
        health_flags=RootStateHealth(REQUIRED_HEALTH_FLAGS),
        position=(0.0, 0.0, 0.72),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.01, 0.0, 0.0),
        angular_velocity=(0.0, 0.01, 0.0),
        covariance_diagonal=(0.01,) * 6,
        calibration_digest=b"c" * 32,
    )


def joint_sample(*, stamp_ns: int | None = None) -> TimedJointSample:
    root = root_packet()
    return TimedJointSample(
        stamp_ns=root.estimate_time_ns if stamp_ns is None else stamp_ns,
        receipt_ns=root.publish_time_ns,
        names=CANONICAL_G1_JOINT_NAMES,
        position=tuple(index / 100 for index in range(29)),
        velocity=tuple(-index / 1000 for index in range(29)),
        sequence=11,
        source_epoch=12,
    )


def test_policy_state_v1_golden_vector_is_440_bytes() -> None:
    payload = serialize_policy_state_v1(golden_packet())
    assert len(payload) == G1_POLICY_STATE_V1_NUM_BYTES == 440
    assert payload[:8] == b"HSPOLI01"
    assert sha256(payload).hexdigest() == GOLDEN_SHA256

    decoded = deserialize_policy_state_v1(payload)
    assert serialize_policy_state_v1(decoded) == payload
    assert decoded.sequence == golden_packet().sequence
    assert decoded.source_epoch == golden_packet().source_epoch
    assert decoded.position == pytest.approx(golden_packet().position)
    assert decoded.quaternion_wxyz == pytest.approx(
        golden_packet().quaternion_wxyz
    )
    assert decoded.linear_velocity == pytest.approx(
        golden_packet().linear_velocity
    )
    assert decoded.angular_velocity == pytest.approx(
        golden_packet().angular_velocity
    )
    assert decoded.covariance_diagonal == pytest.approx(
        golden_packet().covariance_diagonal
    )
    assert decoded.joint_position == pytest.approx(
        golden_packet().joint_position
    )
    assert decoded.joint_velocity == pytest.approx(
        golden_packet().joint_velocity
    )
    assert decoded.calibration_digest == golden_packet().calibration_digest
    assert decoded.joint_mapping_digest == golden_packet().joint_mapping_digest


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("quaternion_wxyz", (2.0, 0.0, 0.0, 0.0), "normalized"),
        ("joint_position", (0.0,) * 28, "29 values"),
        (
            "joint_velocity",
            (0.0,) * 28 + (math.nan,),
            "finite",
        ),
        ("calibration_digest", b"x" * 31, "32 raw bytes"),
        ("joint_mapping_digest", b"y" * 31, "32 raw bytes"),
    ],
)
def test_policy_state_v1_rejects_malformed_semantics(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(PolicyStateProtocolError, match=message):
        serialize_policy_state_v1(replace(golden_packet(), **{field: value}))


def test_build_policy_state_uses_exact_fk_joint_sample() -> None:
    root = root_packet()
    joints = joint_sample()

    packet = build_policy_state_v1(
        root,
        joints,
        joint_mapping_digest=canonical_joint_mapping_digest(),
    )

    assert packet.joint_position == joints.position
    assert packet.joint_velocity == joints.velocity
    assert packet.health_flags == PolicyStateHealth(0xFF)
    assert packet.joint_time_ns == root.joint_time_ns
    assert joints.stamp_ns == root.estimate_time_ns


def test_build_policy_state_rejects_representative_source_sample_substitution() -> None:
    root = root_packet()
    wrong_sample = joint_sample(stamp_ns=root.joint_time_ns)

    with pytest.raises(
        PolicyStateProtocolError,
        match="synchronized root-FK sample",
    ):
        build_policy_state_v1(
            root,
            wrong_sample,
            joint_mapping_digest=canonical_joint_mapping_digest(),
        )


def test_policy_state_decoder_rejects_wrong_magic_and_length() -> None:
    payload = serialize_policy_state_v1(golden_packet())
    with pytest.raises(PolicyStateProtocolError, match="length"):
        deserialize_policy_state_v1(payload[:-1])
    with pytest.raises(PolicyStateProtocolError, match="magic"):
        deserialize_policy_state_v1(b"BADMAGIC" + payload[8:])
