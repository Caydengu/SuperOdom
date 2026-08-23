from __future__ import annotations

import json
from pathlib import Path
import socket
import threading
import time

import pytest

from g1_root_state_bridge.dynamic_capture_io import iter_recorded_datagrams
from g1_root_state_bridge.dynamic_capture_transport import (
    REQUIRED_DYNAMIC_CAPTURE_HEALTH,
    DynamicCapturePacketV1,
    serialize_dynamic_capture_packet,
)
from g1_root_state_bridge.g1_dynamic_capture_recorder import (
    G1DynamicCaptureRecorder,
)
from g1_root_state_bridge.joint_transport import canonical_joint_mapping_digest


def _payload(sequence: int) -> bytes:
    return serialize_dynamic_capture_packet(
        DynamicCapturePacketV1(
            source_epoch=10,
            sequence=sequence,
            robot_stamp_ns=1_000 + sequence,
            source_tick=sequence,
            health_flags=REQUIRED_DYNAMIC_CAPTURE_HEALTH,
            joint_position=tuple(float(index) for index in range(29)),
            joint_velocity=tuple(float(index + 100) for index in range(29)),
            imu_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            imu_gyroscope=(0.125, 0.25, 0.5),
            imu_accelerometer=(0.0, 0.0, 9.8125),
            mapping_digest=canonical_joint_mapping_digest(),
        )
    )


def test_recorder_persists_increasing_packets_and_both_receipt_clocks(
    tmp_path: Path,
) -> None:
    output = tmp_path / "lowstate.hsdyn"
    summary = tmp_path / "summary.json"
    recorder = G1DynamicCaptureRecorder(
        bind_host="127.0.0.1",
        bind_port=0,
        output_path=output,
        summary_path=summary,
        socket_timeout_s=0.01,
    )
    thread = threading.Thread(target=recorder.run, kwargs={"duration_sec": 0.15})
    thread.start()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for sequence in (20, 21, 22):
        sender.sendto(_payload(sequence), ("127.0.0.1", recorder.bound_port))
    sender.close()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    records = list(iter_recorded_datagrams(output))
    assert [record.packet.sequence for record in records] == [20, 21, 22]
    assert all(record.receipt_realtime_ns > 0 for record in records)
    assert all(record.receipt_monotonic_ns > 0 for record in records)
    report = json.loads(summary.read_text())
    assert report["schema"] == "g1_dynamic_capture_summary_v1"
    assert report["accepted"] == 3
    assert report["sequence_gaps"] == 0
    assert report["duplicates_or_reordered"] == 0
    assert report["actuation_topics_created"] == 0


def test_recorder_counts_gap_and_drops_duplicate_without_poisoning_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "lowstate.hsdyn"
    summary = tmp_path / "summary.json"
    recorder = G1DynamicCaptureRecorder(
        bind_host="127.0.0.1",
        bind_port=0,
        output_path=output,
        summary_path=summary,
        socket_timeout_s=0.01,
    )
    thread = threading.Thread(target=recorder.run, kwargs={"duration_sec": 0.15})
    thread.start()
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for sequence in (20, 22, 22):
        sender.sendto(_payload(sequence), ("127.0.0.1", recorder.bound_port))
    sender.close()
    thread.join(timeout=2.0)

    records = list(iter_recorded_datagrams(output))
    assert [record.packet.sequence for record in records] == [20, 22]
    report = json.loads(summary.read_text())
    assert report["accepted"] == 2
    assert report["sequence_gaps"] == 1
    assert report["duplicates_or_reordered"] == 1


def test_recorder_refuses_to_overwrite_output_or_summary(tmp_path: Path) -> None:
    output = tmp_path / "lowstate.hsdyn"
    summary = tmp_path / "summary.json"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        G1DynamicCaptureRecorder(
            bind_host="127.0.0.1",
            bind_port=0,
            output_path=output,
            summary_path=summary,
        )

    output.unlink()
    summary.write_text("{}\n")
    with pytest.raises(FileExistsError):
        G1DynamicCaptureRecorder(
            bind_host="127.0.0.1",
            bind_port=0,
            output_path=output,
            summary_path=summary,
        )


def test_recorder_stop_writes_atomic_summary(tmp_path: Path) -> None:
    output = tmp_path / "lowstate.hsdyn"
    summary = tmp_path / "summary.json"
    recorder = G1DynamicCaptureRecorder(
        bind_host="127.0.0.1",
        bind_port=0,
        output_path=output,
        summary_path=summary,
        socket_timeout_s=0.01,
    )
    thread = threading.Thread(target=recorder.run, kwargs={"duration_sec": 5.0})
    thread.start()
    time.sleep(0.02)
    recorder.stop()
    thread.join(timeout=2.0)

    assert summary.is_file()
    assert not list(tmp_path.glob(".*summary*.tmp"))
    assert json.loads(summary.read_text())["accepted"] == 0
