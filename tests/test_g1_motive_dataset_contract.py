from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_g1_motive_dataset.sh"


def test_launcher_is_capture_only_and_refuses_bad_route() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "actuation_publishers_created\": 0" in source
    assert "g1_dynamic_capture_relay" in source
    assert "live_input_probe.sh" in source
    assert "record_natnet_reference.py" in source
    assert "probe_g1_clock.py" in source
    assert "run_video_recorder" in source
    assert "Robot route is not direct" in source
    assert "/home/unitree/geo-179/G1_localization" in source
    assert "/home/unitree/miniforge3/envs/egonav-deploy/bin/python" in source
    assert "/utlidar/cloud_livox_mid360" in source
    assert "/utlidar/imu_livox_mid360" in source
    assert "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" in source
    assert "--vision-mode" in source
    assert 'source.replace(b"\\xef\\xbb\\xbf", b"")' in source
    assert "g1_lidar_capture" not in source
    assert "ros2 launch livox_ros_driver2" not in source
    assert "rt/lowcmd" not in source
    assert "ChannelPublisher" not in source
