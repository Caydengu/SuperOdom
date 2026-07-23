"""Static safety and evidence contracts for the live root-state shadow."""

from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docker" / "humble-minimal" / "root_state_shadow.sh"


def test_root_state_shadow_is_syntax_valid_and_uses_an_explicit_image():
    assert SCRIPT.is_file()
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    source = SCRIPT.read_text(encoding="utf-8")
    assert "--image" in source
    assert "SUPERODOM_IMAGE" in source
    assert "superodom-root-state-shadow" in source
    assert "duration_sec <= 3600" in source


def test_root_state_shadow_launches_bridge_and_records_only_compact_topics():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "ros2 run g1_root_state_bridge g1-root-state-bridge" in source
    assert "replay_jsonl_path" in source
    assert "/pelvis_state_estimation" in source
    assert "/pelvis_state_bridge/status" in source
    assert "/state_estimation_calibration" in source
    assert "/lidar_correction" in source
    assert "/state_estimation_correction" in source
    assert "strictly_valid" in source
    assert "grep -q packet_published" not in source
    assert "ros2 topic hz" not in source
    assert "setsid ros2 bag record" in source
    assert "setsid ros2 launch super_odometry" in source
    assert "setsid ros2 run g1_root_state_bridge" in source
    assert 'kill -INT -- "-$pid"' in source
    assert "publisher_count()" in source
    assert 'pre_publishers="$(publisher_count /livox/lidar)"' in source
    assert '[[ "$pre_publishers" == "1" ]]' in source
    assert 'active_publishers="$(publisher_count /livox/lidar)"' in source
    assert '[[ "$active_publishers" == "1" ]]' in source
    assert 'kill -TERM -- "-$pid"' in source
    assert 'kill -KILL -- "-$pid"' in source
    assert "$label did not stop after SIGTERM; sending SIGKILL" in source
    assert "term_wait_iterations=25" in source

    record_block = source.split("record_topics=(", maxsplit=1)[1].split(")", maxsplit=1)[0]
    assert "/livox/lidar" not in record_block


def test_reference_configs_disable_forward_prediction():
    for name in ("livox_mid360.yaml", "livox_mid360_gantry.yaml"):
        config = yaml.safe_load(
            (ROOT / "super_odometry" / "config" / name).read_text(encoding="utf-8")
        )
        params = config["/**"]["ros__parameters"]["imu_preintegration_node"]
        assert params["predict_future_secs"] == 0.0
