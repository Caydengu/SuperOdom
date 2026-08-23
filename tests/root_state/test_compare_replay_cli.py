import argparse
import json
from pathlib import Path

from g1_root_state_bridge.compare_replay_cli import compare


def _write(path: Path, treatment: str, offset: float) -> None:
    rows = [
        {"kind": "metadata", "motive_online_input": False},
        *[
            {
                "kind": "pose",
                "treatment": treatment,
                "source_time_ns": index,
                "position_xyz_m": [float(index) + offset, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "source_lowstate_sequence": index,
            }
            for index in range(1, 5)
        ],
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_exact_timestamp_comparison_passes_bounded_drift(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.jsonl"
    reference = tmp_path / "reference.jsonl"
    _write(candidate, "candidate", 0.001)
    _write(reference, "reference", 0.0)
    report = compare(
        argparse.Namespace(
            candidate=candidate,
            candidate_treatment="candidate",
            reference=reference,
            reference_treatment="reference",
            minimum_reference_coverage=0.995,
            maximum_planar_rmse_m=0.01,
            maximum_yaw_rmse_deg=0.5,
        )
    )
    assert report["status"] == "pass"
    assert report["planar_rmse_m"] < 0.002
    assert report["source_joint_sequence_match_fraction"] == 1.0
