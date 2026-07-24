"""Static safety contract for the recorder-free policy-state service."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / "docker" / "humble-minimal" / "policy_state_service.sh"


def test_policy_state_service_is_syntax_valid_and_recorder_free() -> None:
    assert SERVICE.is_file()
    subprocess.run(["bash", "-n", str(SERVICE)], check=True)
    source = SERVICE.read_text(encoding="utf-8")

    assert "--network-interface" in source
    assert "--ros-domain-id" in source
    assert "--output-dir" in source
    assert "--cpuset-cpus" in source
    assert "setsid ros2 launch super_odometry" in source
    assert "setsid ros2 run g1_root_state_bridge" in source
    assert "tcp://*:5576" in source
    assert "policy_state_service_status_v1" in source
    assert "policy_state_service_ready_v1" in source
    assert "deserialize_policy_state_v1" in source
    assert "strictly_valid" in source
    assert "source_epoch" in source
    assert "service_status_max_lines=10000" in source

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
    assert "stop_receipt.json" in source
    assert "active_subscribers" in source
    assert '[[ "$active_subscribers" == "1" ]]' in source
