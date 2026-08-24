from __future__ import annotations

import math

import numpy as np
from g1_root_state_bridge.map_protocol import (
    MAP_CORRECTION_V1_NUM_BYTES,
    REQUIRED_MAP_CORRECTION_HEALTH,
    MapCorrectionPacketV1,
    deserialize_map_correction_v1,
    serialize_map_correction_v1,
)


def test_map_packet_round_trip_is_fixed_and_versioned() -> None:
    pose = np.eye(4)
    angle = math.radians(12.0)
    pose[:2, :2] = (
        (math.cos(angle), -math.sin(angle)),
        (math.sin(angle), math.cos(angle)),
    )
    pose[:3, 3] = (1.0, 2.0, 0.0)
    packet = MapCorrectionPacketV1(
        sequence=2,
        map_epoch=1,
        local_source_epoch=3,
        map_version=4,
        reference_time_ns=10,
        evidence_time_ns=20,
        application_time_ns=30,
        publish_time_ns=30,
        health_flags=REQUIRED_MAP_CORRECTION_HEALTH,
        map_T_local=pose,
        fitness=0.6,
        rmse_m=0.08,
        min_eig=0.03,
        cond_number=50.0,
        covariance_diagonal=(0.01, 0.01, 1.0, 1.0, 1.0, 0.01),
        map_digest=b"m" * 32,
    )
    payload = serialize_map_correction_v1(packet)
    assert len(payload) == MAP_CORRECTION_V1_NUM_BYTES == 176
    decoded = deserialize_map_correction_v1(payload)
    assert decoded.sequence == packet.sequence
    assert decoded.local_source_epoch == packet.local_source_epoch
    assert decoded.map_digest == packet.map_digest
    np.testing.assert_allclose(decoded.map_T_local, pose, atol=1e-6)
