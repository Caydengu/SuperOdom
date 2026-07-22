from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_source_contract_uses_real_output_and_safe_profiles() -> None:
    config = read("super_odometry/config/livox_mid360.yaml")
    gantry = read("super_odometry/config/livox_mid360_gantry.yaml")
    parameters = read("super_odometry/src/parameter/parameter.cpp")
    preintegration = read(
        "super_odometry/src/ImuPreintegration/imuPreintegration_current.cpp"
    )

    assert 'odom_topic: "/state_estimation"' in config
    assert "min_range: 0.5" in config
    assert "min_range: 2.0" in gantry
    assert 'declare_parameter<bool>("use_imu_roll_pitch", false)' in parameters
    assert 'get_parameter("use_imu_roll_pitch").as_bool()' in parameters
    assert "g_world_dir.x() * config_.imuGravity" in preintegration


def test_replay_time_is_explicit_and_live_default_is_safe() -> None:
    launch = read("super_odometry/launch/livox_humanoid.launch.py")
    replay = read("docker/humble-minimal/replay_smoke.sh")

    assert 'DeclareLaunchArgument("use_sim_time"' in launch
    assert 'default_value="false"' in launch
    assert '"use_sim_time": use_sim_time' in launch
    assert "use_sim_time:=true" in replay
    assert "assert_process_sim_time" in replay
    assert '"/proc/$pid/cmdline"' in replay
    assert "ros2 param get" not in replay
    assert "assert_recorder_clock_endpoint" in replay
    assert "Node name: rosbag2_recorder" in replay
    assert "ros2 topic pub" not in replay


def test_dependency_lock_contains_full_revisions() -> None:
    lock = read("docker/humble-minimal/dependency-lock.env")
    expected = {
        "ROS_BASE_IMAGE": "ros:humble-ros-base-jammy@sha256:afb40d6be65331c20a114d4e229a7ef099fed1b17bf6370daee193514b32aa16",
        "SUPERODOM_REVISION": "57a6e233c348372485b7c64ae8238d53c1c7c2ad",
        "LIVOX_DRIVER_REVISION": "6b9356cadf77084619ba406e6a0eb41163b08039",
        "LIVOX_SDK2_REVISION": "6a940156dd7151c3ab6a52442d86bc83613bd11b",
        "GTSAM_REVISION": "4abef9248edc4c49943d8fd8a84c028deb486f4c",
        "SOPHUS_REVISION": "97e71617749a32b83cdd411591ebb2ede9d330f0",
    }
    for key, value in expected.items():
        assert f"{key}={value}" in lock


def test_dockerfile_is_cpu_only_and_pinned() -> None:
    dockerfile = read("docker/humble-minimal/Dockerfile")
    lowered = dockerfile.lower()

    assert "humble-ros-base-jammy" in dockerfile
    assert "rmw-cyclonedds-cpp" in dockerfile
    assert "--packages-select" in dockerfile
    assert "replay_smoke.sh" in dockerfile
    assert "superodom-replay-smoke" in dockerfile
    for forbidden in ("desktop-full", "nvidia", "rviz", "plotjuggler", "x11"):
        assert forbidden not in lowered


def test_entrypoint_enables_nounset_only_after_ros_setup() -> None:
    entrypoint = read("docker/humble-minimal/entrypoint.sh")

    ros_setup = entrypoint.index("source /opt/ros/humble/setup.bash")
    workspace_setup = entrypoint.index('source "$workspace_setup"')
    nounset = entrypoint.index("set -u")
    assert "set -euo pipefail" not in entrypoint[:ros_setup]
    assert nounset > workspace_setup > ros_setup


def test_runtime_wrapper_drops_privilege_and_gpu_requirements(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "output"
    data.mkdir()
    output.mkdir()
    completed = subprocess.run(
        [
            str(ROOT / "docker/humble-minimal/run.sh"),
            "--dry-run",
            "--data-dir",
            str(data),
            "--output-dir",
            str(output),
            "--ros-domain-id",
            "42",
            "--container-name",
            "superodom-contract-test",
            "--network-interface",
            "lo",
            "--",
            "ros2",
            "pkg",
            "list",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "SUPERODOM_IMAGE": "test/superodom:contract"},
    )
    command = completed.stdout
    assert "--network host" in command
    assert "readonly" in command
    assert "--cap-drop ALL" in command
    assert "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" in command
    assert "ROS_DOMAIN_ID=42" in command
    assert "--name superodom-contract-test" in command
    assert "CYCLONEDDS_URI=" in command
    assert "NetworkInterface\\ name=\\\"lo\\\"" in command
    assert "--privileged" not in command
    assert "--gpus" not in command
    assert "--runtime=nvidia" not in command


def test_runtime_wrapper_rejects_invalid_ros_domain_id(tmp_path: Path) -> None:
    data = tmp_path / "data"
    output = tmp_path / "output"
    data.mkdir()
    output.mkdir()

    for invalid in ("-1", "233", "not-a-domain"):
        completed = subprocess.run(
            [
                str(ROOT / "docker/humble-minimal/run.sh"),
                "--dry-run",
                "--data-dir",
                str(data),
                "--output-dir",
                str(output),
                "--ros-domain-id",
                invalid,
                "--",
                "true",
            ],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
        assert completed.returncode == 2
        assert "ROS domain ID must be an integer from 0 through 232" in completed.stderr


def test_live_probe_and_shadow_are_bounded_and_non_actuating(tmp_path: Path) -> None:
    expected = {
        "live_input_probe.sh": ("/livox/lidar", "/livox/imu"),
        "live_shadow.sh": ("/livox/lidar", "/livox/imu", "/state_estimation"),
    }
    forbidden = (
        "/lowcmd",
        "/lowstate",
        "55555",
        "55559",
        "60000",
        "6010",
        "realsense",
        "teleimager",
        "gantry.yaml",
        "use_sim_time:=true",
    )

    for script_name, topics in expected.items():
        script_path = ROOT / "docker/humble-minimal" / script_name
        content = script_path.read_text(encoding="utf-8").lower()
        for topic in topics:
            assert topic in content
        for excluded in forbidden:
            assert excluded not in content
        assert "--ros-domain-id" in content
        assert "timeout" in content
        assert "ros2 bag record" in content
        assert "refusing to overwrite" in content

        output = tmp_path / script_name
        completed = subprocess.run(
            [
                str(script_path),
                "--dry-run",
                "--output-dir",
                str(output),
                "--ros-domain-id",
                "42",
                "--duration-sec",
                "10",
                "--network-interface",
                "lo",
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        assert "ROS_DOMAIN_ID=42" in completed.stdout
        assert "NetworkInterface\\ name=\\\"lo\\\"" in completed.stdout
        for topic in topics:
            assert topic in completed.stdout

    assert "use_sim_time:=false" in read("docker/humble-minimal/live_shadow.sh")
    shadow = read("docker/humble-minimal/live_shadow.sh")
    assert "record_lidar=false" in shadow
    assert "record_topics=(" in shadow
    for diagnostic_topic in (
        "/livox/imu",
        "/state_estimation",
        "/laser_odometry",
        "/super_odometry_stats",
        "/state_estimation_health",
        "/prediction_source",
    ):
        assert diagnostic_topic in shadow
    assert 'if [[ "$record_lidar" == true ]]' in shadow
    assert "live_superodometry_shadow_output_only" in shadow


def test_shell_scripts_are_syntactically_valid() -> None:
    scripts = [
        ROOT / "docker/humble-minimal/build_image.sh",
        ROOT / "docker/humble-minimal/run.sh",
        ROOT / "docker/humble-minimal/entrypoint.sh",
        ROOT / "docker/humble-minimal/replay_smoke.sh",
        ROOT / "docker/humble-minimal/live_input_probe.sh",
        ROOT / "docker/humble-minimal/live_shadow.sh",
        ROOT / "scripts/test_minimal_container.sh",
    ]
    for script in scripts:
        subprocess.run(["bash", "-n", str(script)], check=True)
