import socket
import time

from g1_root_state_bridge.dynamic_capture_transport import (
    REQUIRED_DYNAMIC_CAPTURE_HEALTH,
    DynamicCapturePacketV1,
    serialize_dynamic_capture_packet,
)
from g1_root_state_bridge.joint_transport import canonical_joint_mapping_digest
from g1_root_state_bridge.udp_lowstate import DynamicLowStateReceiver


def test_udp_receiver_accepts_only_valid_read_only_packet() -> None:
    received = []
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    available_port = probe.getsockname()[1]
    probe.close()
    receiver = DynamicLowStateReceiver(
        bind_host="127.0.0.1",
        bind_port=available_port,
        callback=lambda packet, stamp: received.append((packet, stamp)),
    )
    port = receiver.socket.getsockname()[1]
    receiver.start()
    packet = DynamicCapturePacketV1(
        source_epoch=1,
        sequence=2,
        robot_stamp_ns=time.time_ns(),
        source_tick=3,
        health_flags=REQUIRED_DYNAMIC_CAPTURE_HEALTH,
        joint_position=(0.0,) * 29,
        joint_velocity=(0.0,) * 29,
        imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        imu_gyroscope=(0.0, 0.0, 0.0),
        imu_accelerometer=(0.0, 0.0, 9.81),
        mapping_digest=canonical_joint_mapping_digest(),
    )
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.sendto(serialize_dynamic_capture_packet(packet), ("127.0.0.1", port))
    deadline = time.monotonic() + 1.0
    while not received and time.monotonic() < deadline:
        time.sleep(0.01)
    sender.close()
    receiver.close()
    assert len(received) == 1
    assert serialize_dynamic_capture_packet(received[0][0]) == (
        serialize_dynamic_capture_packet(packet)
    )
    assert receiver.invalid == 0
