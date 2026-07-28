from __future__ import annotations

import socket
import sys
from types import SimpleNamespace

import pytest

from g1_root_state_bridge.dynamic_capture_transport import (
    deserialize_dynamic_capture_packet,
)


def _install_subscriber_only_sdk(monkeypatch):
    calls: list[tuple[object, ...]] = []

    class LowState:
        pass

    class Subscriber:
        def __init__(self, topic, message_type):
            calls.append(("subscriber", topic, message_type))

        def Init(self, callback, queue_len):  # noqa: N802
            calls.append(("subscriber_init", queue_len))
            self.callback = callback

        def Close(self):  # noqa: N802
            calls.append(("subscriber_close",))

    channel = SimpleNamespace(
        ChannelFactoryInitialize=lambda domain, interface: calls.append(
            ("factory", domain, interface)
        ),
        ChannelSubscriber=Subscriber,
        ChannelPublisher=lambda *_args: pytest.fail(
            "dynamic capture relay must never construct a publisher"
        ),
    )
    idl = SimpleNamespace(LowState_=LowState)
    monkeypatch.setitem(sys.modules, "unitree_sdk2py.core.channel", channel)
    monkeypatch.setitem(
        sys.modules,
        "unitree_sdk2py.idl.unitree_hg.msg.dds_",
        idl,
    )
    return calls


def _state() -> SimpleNamespace:
    return SimpleNamespace(
        motor_state=[
            SimpleNamespace(q=float(index), dq=float(index + 100))
            for index in range(29)
        ],
        imu_state=SimpleNamespace(
            quaternion=[1.0, 0.0, 0.0, 0.0],
            gyroscope=[0.125, 0.25, 0.5],
            accelerometer=[0.0, 0.0, 9.8125],
        ),
        tick=1234,
    )


def test_dynamic_relay_constructs_only_subscriber_and_emits_typed_imu_packet(
    monkeypatch,
) -> None:
    calls = _install_subscriber_only_sdk(monkeypatch)
    from g1_root_state_bridge.g1_dynamic_capture_relay import (
        G1DynamicCaptureRelay,
    )

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(1.0)
    relay = G1DynamicCaptureRelay(
        target_host="127.0.0.1",
        target_port=receiver.getsockname()[1],
        network_interface="robot-nic",
        domain_id=0,
        max_rate_hz=2_000.0,
    )

    relay.callback(_state())
    payload, _address = receiver.recvfrom(4096)
    packet = deserialize_dynamic_capture_packet(payload)
    relay.close()
    receiver.close()

    assert packet.joint_position == tuple(float(index) for index in range(29))
    assert packet.joint_velocity == tuple(float(index + 100) for index in range(29))
    assert packet.imu_quaternion_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert packet.imu_gyroscope == (0.125, 0.25, 0.5)
    assert packet.imu_accelerometer == (0.0, 0.0, 9.8125)
    assert packet.source_tick == 1234
    assert calls[0] == ("factory", 0, "robot-nic")
    assert calls[1][0:2] == ("subscriber", "rt/lowstate")
    assert not hasattr(relay, "send_low_command")
    assert relay.summary()["actuation_topics_created"] == 0


def test_dynamic_relay_rejects_missing_imu_without_zero_fallback(
    monkeypatch,
) -> None:
    _install_subscriber_only_sdk(monkeypatch)
    from g1_root_state_bridge.g1_dynamic_capture_relay import (
        G1DynamicCaptureRelay,
    )

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(0.02)
    relay = G1DynamicCaptureRelay(
        target_host="127.0.0.1",
        target_port=receiver.getsockname()[1],
        network_interface="robot-nic",
        domain_id=0,
        max_rate_hz=2_000.0,
    )
    malformed = _state()
    malformed.imu_state = SimpleNamespace(
        quaternion=[1.0, 0.0, 0.0, 0.0],
        gyroscope=[0.0, 0.0, 0.0],
    )

    relay.callback(malformed)

    with pytest.raises(TimeoutError):
        receiver.recvfrom(4096)
    assert relay.summary()["invalid_imu"] == 1
    relay.close()
    receiver.close()


def test_dynamic_relay_source_contains_no_command_capability() -> None:
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "g1_root_state_bridge"
        / "g1_root_state_bridge"
        / "g1_dynamic_capture_relay.py"
    ).read_text(encoding="utf-8")

    assert 'ChannelSubscriber("rt/lowstate", LowState_)' in source
    for forbidden in (
        "ChannelPublisher",
        "LowCmd_",
        "rt/lowcmd",
        "send_low_command",
    ):
        assert forbidden not in source
