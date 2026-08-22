from __future__ import annotations

import json

import pytest
from g1_root_state_bridge.amo_treatments import AmoTreatmentError, load_odometry_track


def _record(source_time_ns: int, receipt_time_ns: int) -> dict[str, object]:
    return {
        "kind": "odometry",
        "child_frame_id": "sensor",
        "source_time_ns": source_time_ns,
        "receipt_time_ns": receipt_time_ns,
    }


def test_load_odometry_rejects_isolated_stale_message_without_reordering(
    tmp_path,
) -> None:
    path = tmp_path / "track.jsonl"
    rows = [
        {"kind": "metadata"},
        _record(100, 1_000),
        _record(200, 2_000),
        _record(190, 2_000),
        _record(300, 3_000),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    loaded = load_odometry_track(path, maximum_stale_fraction=0.5)
    assert [row["source_time_ns"] for row in loaded.records] == [100, 200, 300]
    assert loaded.health["rejected_stale_count"] == 1
    assert loaded.rejected_stale_records[0]["rollback_ns"] == 10


def test_load_odometry_fails_closed_on_persistent_reset(tmp_path) -> None:
    path = tmp_path / "track.jsonl"
    rows = [
        {"kind": "metadata"},
        _record(100, 1_000),
        _record(200, 2_000),
        _record(10, 3_000),
        _record(20, 4_000),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(AmoTreatmentError, match="possible persistent estimator reset"):
        load_odometry_track(path, maximum_stale_fraction=0.1)
