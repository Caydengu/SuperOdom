from __future__ import annotations

import json
import math
import sys
import threading
import time

import numpy as np
from g1_root_state_bridge.map_protocol import (
    MAP_CORRECTION_V1_NUM_BYTES,
    REQUIRED_MAP_CORRECTION_HEALTH,
    MapCorrectionPacketV1,
    deserialize_map_correction_v1,
    serialize_map_correction_v1,
)
from g1_root_state_bridge.map_readiness import (
    main as map_readiness_main,
)
from g1_root_state_bridge.map_readiness import (
    readiness_rejection_reason,
)


def _packet(now_ns: int = 30) -> MapCorrectionPacketV1:
    pose = np.eye(4)
    angle = math.radians(12.0)
    pose[:2, :2] = (
        (math.cos(angle), -math.sin(angle)),
        (math.sin(angle), math.cos(angle)),
    )
    pose[:3, 3] = (1.0, 2.0, 0.0)
    return MapCorrectionPacketV1(
        sequence=2,
        map_epoch=1,
        local_source_epoch=3,
        map_version=4,
        reference_time_ns=now_ns - 20,
        evidence_time_ns=now_ns - 10,
        application_time_ns=now_ns,
        publish_time_ns=now_ns,
        health_flags=REQUIRED_MAP_CORRECTION_HEALTH,
        map_T_local=pose,
        fitness=0.6,
        rmse_m=0.08,
        min_eig=0.03,
        cond_number=50.0,
        covariance_diagonal=(0.01, 0.01, 1.0, 1.0, 1.0, 0.01),
        map_digest=b"m" * 32,
    )


def test_map_packet_round_trip_is_fixed_and_versioned() -> None:
    packet = _packet()
    payload = serialize_map_correction_v1(packet)
    assert len(payload) == MAP_CORRECTION_V1_NUM_BYTES == 176
    decoded = deserialize_map_correction_v1(payload)
    assert decoded.sequence == packet.sequence
    assert decoded.local_source_epoch == packet.local_source_epoch
    assert decoded.map_digest == packet.map_digest
    np.testing.assert_allclose(decoded.map_T_local, packet.map_T_local, atol=1e-6)


def test_map_readiness_is_identity_epoch_health_and_freshness_bound() -> None:
    packet = _packet()
    arguments = {
        "expected_map_digest": packet.map_digest,
        "expected_local_source_epoch": packet.local_source_epoch,
        "now_ns": packet.publish_time_ns + 10_000_000,
        "maximum_age_ns": 500_000_000,
    }
    assert readiness_rejection_reason(packet, **arguments) is None
    assert (
        readiness_rejection_reason(
            packet,
            **{**arguments, "expected_map_digest": b"x" * 32},
        )
        == "map_digest"
    )
    assert (
        readiness_rejection_reason(
            packet,
            **{
                **arguments,
                "expected_local_source_epoch": packet.local_source_epoch + 1,
            },
        )
        == "local_source_epoch"
    )
    assert (
        readiness_rejection_reason(
            packet,
            **{**arguments, "now_ns": packet.publish_time_ns + 500_000_001},
        )
        == "map_packet_stale"
    )


def test_map_readiness_cli_accepts_one_loopback_packet(tmp_path, monkeypatch) -> None:
    import zmq

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")

    def publish() -> None:
        time.sleep(0.2)
        for _ in range(20):
            publisher.send(serialize_map_correction_v1(_packet(time.time_ns())))
            time.sleep(0.02)

    thread = threading.Thread(target=publish)
    thread.start()
    output = tmp_path / "map-readiness.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "g1-wait-map-correction",
            "--endpoint",
            f"tcp://127.0.0.1:{port}",
            "--map-sha256",
            (b"m" * 32).hex(),
            "--local-source-epoch",
            "3",
            "--timeout-sec",
            "2",
            "--output",
            str(output),
        ],
    )
    try:
        assert map_readiness_main() == 0
    finally:
        thread.join()
        publisher.close(linger=0)
        context.term()
    report = json.loads(output.read_text())
    assert report["status"] == "ready"
    assert report["map_digest"] == (b"m" * 32).hex()
    assert report["command_capability"] == "structurally_unavailable"
