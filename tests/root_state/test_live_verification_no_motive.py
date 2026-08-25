from __future__ import annotations

from pathlib import Path

from scripts import validate_g1_kiss_live_verification as validator


def test_missing_optional_artifact_is_a_failed_check_not_a_validator_crash(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "robot-vlm-real-backend.json"
    result = validator.load_optional_json(artifact)
    assert result["status"] == "missing_or_invalid"
    assert result["artifact"] == str(artifact)
    assert "FileNotFoundError" in result["error"]


def test_integrated_launcher_accepts_only_the_expected_live_stack_stop_signal() -> None:
    healthy = {
        "live_stack": 143,
        "motive": 0,
        "lowstate_recorder": 0,
        "lowstate_relay": 0,
        "rosbag": 0,
        "robot_vlm_real_backend": 0,
    }
    assert validator.process_statuses_ok(healthy, integrated_robot_vlm=True)
    assert not validator.process_statuses_ok(healthy, integrated_robot_vlm=False)
    assert not validator.process_statuses_ok(
        {**healthy, "robot_vlm_real_backend": 127},
        integrated_robot_vlm=True,
    )


def test_no_motive_mode_qualifies_operation_without_claiming_absolute_accuracy(
    monkeypatch,
) -> None:
    def fake_load_json(path: Path):
        if path.name == "manifest.json":
            return {
                "duration_sec": 1,
                "capture_class": "stationary",
                "motive_mode": "disabled",
                "integrated_robot_vlm": False,
            }
        if path.name == "process_status.json":
            return {
                "live_stack": 0,
                "motive": 0,
                "lowstate_recorder": 0,
                "lowstate_relay": 0,
                "rosbag": 0,
                "robot_vlm_real_backend": 0,
            }
        if path.parent.name == "motive":
            raise AssertionError("disabled mode must not load a Motive artifact")
        if path.parent.name == "lowstate":
            return {
                "accepted": 300,
                "invalid": 0,
                "sequence_gaps": 0,
                "duplicates_or_reordered": 0,
                "actuation_topics_created": 0,
            }
        raise AssertionError(f"unexpected JSON load: {path}")

    monkeypatch.setattr(validator, "load_json", fake_load_json)
    monkeypatch.setattr(
        validator,
        "bag_counts",
        lambda _path: {topic: 300 for topic in validator.TOPICS},
    )
    monkeypatch.setattr(
        validator,
        "last_live_status",
        lambda _path: {
            "active": True,
            "command_capability": "structurally_unavailable",
            "lowstate_udp_invalid": 0,
            "stats": {
                "published": 100,
                "lidar": 100,
                "queue_replaced": 0,
                "imu_wait_timeouts": 0,
                "joint_wait_timeouts": 0,
                "transport_rejected": 0,
            },
            "stage_runtime_ms": {"total": {"p95": 10.0, "p99": 15.0}},
            "pose_age_ms": {"p95": 20.0},
        },
    )

    result = validator.validate(Path("/tmp/no-motive-live-run"))
    assert result["status"] == "pass"
    assert result["checks"]["motive"]["status"] == "disabled"
    assert result["absolute_map_pose_accuracy_evaluated"] is False
    assert result["external_ground_truth_recorded"] is False
    assert "without external ground truth" in result["evidence_class"]
