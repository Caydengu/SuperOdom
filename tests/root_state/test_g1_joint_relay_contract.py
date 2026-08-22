from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_g1_relay_is_typed_read_only_and_stamps_before_networking() -> None:
    source = (
        ROOT / "g1_root_state_bridge" / "g1_root_state_bridge" / "g1_joint_relay.py"
    ).read_text(encoding="utf-8")

    assert "unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_" in source
    assert 'ChannelSubscriber("rt/lowstate", LowState_)' in source
    assert "time.time_ns()" in source
    assert "socket.SOCK_DGRAM" in source
    assert "serialize_joint_packet" in source
    assert "serialize_root_imu_packet" in source
    assert "imu.quaternion" in source
    assert "imu.gyroscope" in source
    for forbidden in (
        "ChannelPublisher",
        "LowCmd_",
        "rt/lowcmd",
        "rclpy",
        "deserialize_cdr",
        "struct.unpack_from",
    ):
        assert forbidden not in source
