import argparse
import json
from pathlib import Path

import pytest

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
            pose_comparison="raw",
        )
    )
    assert report["status"] == "pass"
    assert report["planar_rmse_m"] < 0.002
    assert report["source_joint_sequence_match_fraction"] == 1.0


def test_initial_relative_comparison_removes_only_local_origin_offset(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.jsonl"
    reference = tmp_path / "reference.jsonl"
    _write(candidate, "candidate", 0.1)
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
            pose_comparison="initial-relative",
        )
    )
    assert report["status"] == "pass"
    assert report["raw_planar_rmse_m"] == pytest.approx(0.1)
    assert report["planar_rmse_m"] == pytest.approx(0.0)


def test_comparison_uses_lowstate_sequence_when_source_clocks_are_remapped(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.jsonl"
    reference = tmp_path / "reference.jsonl"
    _write(candidate, "candidate", 0.001)
    _write(reference, "reference", 0.0)
    rows = [json.loads(line) for line in candidate.read_text().splitlines()]
    for row in rows:
        if row.get("kind") == "pose":
            row["source_time_ns"] += 2_000_000
    candidate.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = compare(
        argparse.Namespace(
            candidate=candidate,
            candidate_treatment="candidate",
            reference=reference,
            reference_treatment="reference",
            minimum_reference_coverage=0.995,
            maximum_planar_rmse_m=0.01,
            maximum_yaw_rmse_deg=0.5,
            pose_comparison="raw",
        )
    )
    assert report["status"] == "pass"
    assert report["match_key"] == "source_lowstate_sequence"
    assert report["exact_timestamp_overlap"] == 0
    assert report["matched_timestamp_delta_ms"]["p95"] == pytest.approx(2.0)
