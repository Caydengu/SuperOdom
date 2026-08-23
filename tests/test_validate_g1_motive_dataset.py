import json
from pathlib import Path

from scripts.validate_g1_motive_dataset import check_dataset


def write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_g1_4123_native_topics_pass_without_optional_vision(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {
            "lidar_topic": "/utlidar/cloud_livox_mid360",
            "imu_topic": "/utlidar/imu_livox_mid360",
            "vision_enabled": False,
        },
    )
    write_json(
        tmp_path / "motive" / "summary.json",
        {"connected": True, "frames_written": 1200, "tracking_coverage": 1.0},
    )
    write_json(
        tmp_path / "lowstate" / "summary.json",
        {"accepted": 10000, "invalid": 0, "sequence_gaps": 0, "accepted_rate_hz": 1000.0},
    )
    metadata = tmp_path / "lidar" / "data" / "live_input_probe" / "metadata.yaml"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        """topics_with_message_count:
  - topic_metadata:
      name: /utlidar/cloud_livox_mid360
    message_count: 100
  - topic_metadata:
      name: /utlidar/imu_livox_mid360
    message_count: 2000
""",
        encoding="utf-8",
    )
    write_json(
        tmp_path / "process_status.json",
        {"motive": 0, "lowstate_recorder": 0, "lowstate_relay": 0, "lidar": 0, "vision": 0},
    )

    result = check_dataset(tmp_path)

    assert result["status"] == "pass"
    assert result["checks"]["rgbd"]["required"] is False
    assert result["checks"]["lidar_and_livox_imu"]["message_counts"] == {
        "/utlidar/cloud_livox_mid360": 100,
        "/utlidar/imu_livox_mid360": 2000,
    }
