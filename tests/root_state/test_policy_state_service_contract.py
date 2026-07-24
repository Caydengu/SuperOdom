"""Static safety contract for the recorder-free policy-state service."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / "docker" / "humble-minimal" / "policy_state_service.sh"
OSLO_LIVOX_CONFIG = (
    ROOT / "super_odometry" / "config" / "livox" / "MID360_oslo.json"
)


def test_policy_state_service_is_syntax_valid_and_recorder_free() -> None:
    assert SERVICE.is_file()
    subprocess.run(["bash", "-n", str(SERVICE)], check=True)
    source = SERVICE.read_text(encoding="utf-8")

    assert "--network-interface" in source
    assert "--ros-domain-id" in source
    assert "--output-dir" in source
    assert "--cpuset-cpus" in source
    assert "--livox-config" in source
    assert "setsid ros2 run livox_ros_driver2 livox_ros_driver2_node" in source
    assert "setsid ros2 launch super_odometry" in source
    assert "setsid ros2 run g1_root_state_bridge" in source
    assert "tcp://*:5576" in source
    assert "policy_state_service_status_v1" in source
    assert "policy_state_service_ready_v1" in source
    assert "deserialize_policy_state_v1" in source
    assert "strictly_valid" in source
    assert "source_epoch" in source
    assert "service_status_max_lines=10000" in source
    assert "livox_input_preflight_v1" in source
    assert "/output/livox_input_preflight.json" in source

    for forbidden in (
        "ros2 bag record",
        "replay_jsonl_path:=",
        "/livox/lidar" + " --output",
        "killall",
        "pkill",
    ):
        assert forbidden not in source


def test_policy_state_service_owns_only_its_child_process_groups() -> None:
    source = SERVICE.read_text(encoding="utf-8")
    assert 'kill -INT -- "-$pid"' in source
    assert 'kill -TERM -- "-$pid"' in source
    assert 'kill -KILL -- "-$pid"' in source
    assert "launch_pid=" in source
    assert "bridge_pid=" in source
    assert "livox_pid=" in source
    assert 'stop_process "$livox_pid" livox-driver' in source
    assert "stop_receipt.json" in source
    assert "active_subscribers" in source
    assert '[[ "$active_subscribers" == "1" ]]' in source


def test_oslo_livox_config_routes_mid360_directly_to_the_robot_nic() -> None:
    source = OSLO_LIVOX_CONFIG.read_text(encoding="utf-8")

    assert source.count('"192.168.123.11"') == 4
    assert '"ip": "192.168.123.120"' in source
    assert '"roll": 180.0' in source
    assert '"imu_data_port": 56401' in source
