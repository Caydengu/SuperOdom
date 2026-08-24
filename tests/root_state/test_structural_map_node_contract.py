from __future__ import annotations

from g1_root_state_bridge.protocol import (
    REQUIRED_ROOT_FUSION_FLAGS,
    RootStatePacketV2,
)
from g1_root_state_bridge.structural_map_localization import StructuralMapConfig
from g1_root_state_bridge.structural_map_node import (
    RootEvidenceSnapshot,
    root_evidence_rejection_reason,
)


def root_packet(
    *, estimate_time_ns: int, health=REQUIRED_ROOT_FUSION_FLAGS
) -> RootStatePacketV2:
    return RootStatePacketV2(
        sequence=1,
        source_epoch=2,
        estimate_time_ns=estimate_time_ns,
        publish_time_ns=estimate_time_ns + 1,
        correction_time_ns=estimate_time_ns,
        joint_time_ns=estimate_time_ns,
        joint_sync_gap_ns=0,
        health_flags=health,
        position=(0.0, 0.0, 0.8),
        quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        covariance_diagonal=(0.01,) * 6,
        calibration_digest=b"c" * 32,
    )


def test_root_evidence_requires_caught_up_healthy_common_clock_packet() -> None:
    config = StructuralMapConfig()
    evidence = 1_000_000_000
    assert (
        root_evidence_rejection_reason(
            None,
            evidence_time_ns=evidence,
            now_ns=evidence + 10_000_000,
            config=config,
        )
        == "root_state_unavailable"
    )
    behind = RootEvidenceSnapshot(root_packet(estimate_time_ns=evidence - 1), evidence)
    assert (
        root_evidence_rejection_reason(
            behind,
            evidence_time_ns=evidence,
            now_ns=evidence + 10_000_000,
            config=config,
        )
        == "root_state_not_caught_up"
    )
    healthy = RootEvidenceSnapshot(root_packet(estimate_time_ns=evidence), evidence)
    assert (
        root_evidence_rejection_reason(
            healthy,
            evidence_time_ns=evidence,
            now_ns=evidence + 10_000_000,
            config=config,
        )
        is None
    )


def test_root_evidence_rejects_stale_input_before_registration() -> None:
    config = StructuralMapConfig()
    evidence = 1_000_000_000
    snapshot = RootEvidenceSnapshot(root_packet(estimate_time_ns=evidence), evidence)
    assert (
        root_evidence_rejection_reason(
            snapshot,
            evidence_time_ns=evidence,
            now_ns=evidence + config.maximum_input_age_ns + 1,
            config=config,
        )
        == "map_evidence_stale_before_registration"
    )
