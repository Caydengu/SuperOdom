from pathlib import Path

import numpy as np
from g1_root_state_bridge.live_node import StartupGyroBiasEstimator

ROOT = Path(__file__).resolve().parents[2]


def test_startup_gyro_bias_requires_a_stationary_five_second_window() -> None:
    estimator = StartupGyroBiasEstimator(
        duration_sec=5.0,
        maximum_axis_std_radps=0.02,
        maximum_leg_rms_radps=0.05,
    )
    bias = np.asarray((0.025, -0.021, -0.015), dtype=np.float64)
    result = None
    for index in range(1_001):
        offset_ns = index * 5_000_000
        estimator.observe_leg(
            1_000_000_000_000 + offset_ns,
            np.zeros(29, dtype=np.float64),
        )
        noise = 0.001 * np.sin(index * 0.1)
        result = estimator.observe_imu(
            2_000_000_000_000 + offset_ns,
            bias + np.asarray((noise, -noise, noise)),
        )
    assert result is not None
    np.testing.assert_allclose(result.bias_radps, bias, atol=1e-4)
    assert result.imu_sample_count >= 1_000
    assert result.joint_sample_count >= 1_000


def test_startup_gyro_bias_rejects_a_moving_leg_window() -> None:
    estimator = StartupGyroBiasEstimator(
        duration_sec=5.0,
        maximum_axis_std_radps=0.02,
        maximum_leg_rms_radps=0.05,
    )
    result = None
    for index in range(1_001):
        offset_ns = index * 5_000_000
        estimator.observe_leg(
            1_000_000_000_000 + offset_ns,
            np.full(29, 0.1, dtype=np.float64),
        )
        result = estimator.observe_imu(
            2_000_000_000_000 + offset_ns,
            np.asarray((0.025, -0.021, -0.015)),
        )
    assert result is None


def test_live_node_uses_deployed_topics_and_has_no_unitree_command_surface() -> None:
    source = (
        ROOT / "g1_root_state_bridge/g1_root_state_bridge/live_node.py"
    ).read_text()
    assert "/utlidar/cloud_livox_mid360" in source
    assert "/utlidar/imu_livox_mid360" in source
    assert "/g1/localization/cloud_registered" in source
    assert "tcp://*:5575" in source
    assert "ChannelPublisher" not in source
    assert "rt/lowcmd" not in source
    assert "sportmodestate" not in source.lower()
    assert '"structurally_unavailable"' in source


def test_setup_exposes_live_and_replay_entrypoints() -> None:
    setup = (ROOT / "g1_root_state_bridge/setup.py").read_text()
    assert "g1-kiss-live-localization" in setup
    assert "g1-kiss-localization-replay" in setup
    assert "g1-wait-map-correction" in setup
    assert "g1_root_state_bridge.bridge_node" not in setup


def test_container_entrypoint_sources_ros_with_nounset_disabled() -> None:
    source = (ROOT / "docker/kiss-live/entrypoint.sh").read_text()
    disable_nounset = source.index("set +u")
    source_ros = source.index("source /opt/ros/humble/setup.bash")
    source_workspace = source.index("source /opt/g1_localization_ws/install/setup.bash")
    restore_nounset = source.index("set -u", source_workspace)
    assert disable_nounset < source_ros < source_workspace < restore_nounset
    assert (
        'export PATH="/opt/g1_localization_ws/install/lib/g1_root_state_bridge:$PATH"'
        in source
    )


def test_live_node_discards_cross_epoch_scans_and_survives_zmq_backpressure() -> None:
    source = (
        ROOT / "g1_root_state_bridge/g1_root_state_bridge/live_node.py"
    ).read_text()
    assert "queued_epoch != self.pipeline_epoch" in source
    assert "except zmq.Again" in source
    assert 'self.stats["transport_rejected"] += 1' in source
    assert '"rejection_reasons": self.rejection_reasons' in source
    assert "scan_pipeline:{error}" in source
    assert "except ImuCoveragePending:" in source
    assert "except JointCoveragePending:" in source
    assert "coverage wait timed out" in source
    assert "--source-coverage-wait-ms" in source
    assert "assert_scan_sources_ready" in source
    assert '"stage_runtime_ms"' in source
    assert '"pose_age_ms"' in source
    assert "--maximum-imu-gap-ms" in source
    assert "--maximum-imu-bridge-gap-ms" in source
    assert 'self.stats["imu_gap_bridges"] += 1' in source


def test_live_node_matches_the_recorded_reliable_sensor_publishers() -> None:
    source = (
        ROOT / "g1_root_state_bridge/g1_root_state_bridge/live_node.py"
    ).read_text()
    assert "imu_qos = QoSProfile" in source
    assert "lidar_qos = QoSProfile" in source
    assert source.count("reliability=ReliabilityPolicy.RELIABLE") >= 2
    assert "depth=64" in source


def test_live_node_shutdown_is_idempotent_after_sigint() -> None:
    source = (
        ROOT / "g1_root_state_bridge/g1_root_state_bridge/live_node.py"
    ).read_text()
    assert "except KeyboardInterrupt:" in source
    assert "if rclpy.ok():" in source
    assert "rclpy.shutdown()" in source


def test_live_node_publishes_map_evidence_in_the_kiss_local_frame() -> None:
    source = (
        ROOT / "g1_root_state_bridge/g1_root_state_bridge/live_node.py"
    ).read_text()
    assert "output.registered_points_local_xyz_m" in source
    assert "output.registered_cloud_xyz32" in source
    assert 'cloud.header.frame_id = "kiss_local"' in source
    assert "packet.estimate_time_ns" in source
    assert "PointField.FLOAT32" in source


def test_transient_clock_delay_drops_sample_without_resetting_local_map() -> None:
    source = (
        ROOT / "g1_root_state_bridge/g1_root_state_bridge/live_node.py"
    ).read_text()
    assert "classify_clock_estimate" in source
    assert "ClockSampleDisposition.RESET_LANE" in source
    assert "ClockSampleDisposition.ACCEPT" in source
    assert "clock_fit_unhealthy" in source


def test_passive_launcher_names_no_policy_or_command_channel() -> None:
    source = (ROOT / "scripts/run_g1_kiss_live_stack.sh").read_text()
    assert "g1_dynamic_capture_relay" in source
    assert "run_live.sh" in source
    assert "rt/lowcmd" not in source
    assert "run_amo" not in source
    assert "command_capability=structurally_unavailable" in source
    assert "/home/unitree/miniforge3/envs/egonav-deploy/bin/python" in source


def test_passive_launcher_cleanup_owns_both_producer_and_relay() -> None:
    source = (ROOT / "scripts/run_g1_kiss_live_stack.sh").read_text()
    assert "producer_pid=$!" in source
    assert 'for pid in "$producer_pid" "$relay_pid"' in source
    assert 'docker stop --time 3 "$container_name"' in source
    assert "--container-name $container_name" in source
    assert "--root-state-port $root_state_port" in source
    assert "remote_pid_file" in source
    assert "[g]1_dynamic_capture_relay.*--target-port $lowstate_port" in source
    assert "trap 'exit 143' TERM" in source


def test_structural_map_shadow_launcher_is_digest_bound_and_command_incapable() -> None:
    source = (ROOT / "scripts/run_g1_structural_map_shadow.sh").read_text()
    assert "sha256sum" in source
    assert "map_xy_all_5cm" in source
    assert "g1-structural-map-localization" in source
    assert "tcp://127.0.0.1:$root_state_port" in source
    assert "tcp://*:$map_correction_port" in source
    assert "--read-only" in source
    assert "command_capability=structurally_unavailable" in source
    assert "rt/lowcmd" not in source
    assert "run_amo" not in source


def test_layered_qualification_runs_both_lanes_and_actual_robot_vlm_probe() -> None:
    source = (ROOT / "scripts/run_g1_layered_localization_qualification.sh").read_text()
    assert "run_g1_kiss_live_stack.sh" in source
    assert "run_g1_structural_map_shadow.sh" in source
    assert "g1-wait-map-correction" in source
    assert "--network host" in source
    assert "--read-only" in source
    assert "probe_layered_localization.py" in source
    assert "--expected-map present" in source
    assert "command_capability" in source
    assert "structurally_unavailable" in source
    assert "hardware_actuation_clearance" in source
    assert "rt/lowcmd" not in source
    assert "run_amo" not in source


def test_live_verification_records_proven_inputs_outputs_and_motive_truth() -> None:
    source = (ROOT / "scripts/run_g1_kiss_live_verification.sh").read_text()
    assert "--motive-mode" in source
    assert '"$motive_mode" == required' in source
    assert '"$motive_mode" == unmapped' in source
    assert '"$motive_mode" == disabled' in source
    assert "record_natnet_reference.py" in source
    assert "g1_dynamic_capture_recorder" in source
    assert "/utlidar/cloud_livox_mid360" in source
    assert "/utlidar/imu_livox_mid360" in source
    assert "/g1/localization/pelvis_odom" in source
    assert "/g1/localization/cloud_registered" in source
    assert "G1_PELVIS_F_4123" in source
    assert "rigid_body_id=42" in source
    assert "actuation_publishers_created" in source
    assert "wait_for_root_state.py" in source
    assert "tcp://127.0.0.1:5575" in source
    assert '--endpoint "tcp://127.0.0.1:$root_state_port"' in source
    assert "READY FOR OPERATOR-CONTROLLED AMO" in source
    assert 'robot_python=/home/unitree/miniforge3/envs/egonav-deploy/bin/python' in source
    assert '--robot-python "$robot_python"' in source
    assert '"robot_python": "$robot_python"' in source
    assert "mark_runtime_failed map_initialization" in source
    assert "no accepted digest-bound map correction" in source
    assert 'robot_vlm_python="$robot_vlm_repo/.venv/bin/python"' in source
    assert 'PYTHONPATH=src "$robot_vlm_python"' in source
    assert "PYTHONPATH=src uv run" not in source
    assert "[1/5] Verifying SSH access" in source
    assert "[2/5] Binding the physical robot fingerprint" in source
    assert "[5/5] Passive relay staged" in source
    assert "preflight_failed" in source
    assert "mark_runtime_failed validation" in source
    assert "capture validation failed" in source
    assert "wrong_robot_identity" in source
    assert "clock_probe" in source
    assert "infrastructure_failed" in source
    assert "--timeout-sec 5" in source
    assert 'value["robot_machine_id_sha256"]' in source
    assert "ConnectTimeout=5" in source
    assert source.index("wait_for_root_state.py") < source.index(
        "record_natnet_reference.py"
    )
    assert "rt/lowcmd" not in source
    assert "run_amo" not in source


def test_fieldbay_wrapper_pins_the_complete_g1_4123_treatment() -> None:
    source = (ROOT / "scripts/run_g1_4123_robot_vlm_fieldbay.sh").read_text()
    assert "run_g1_kiss_live_verification.sh" in source
    assert "--integrated-robot-vlm" in source
    assert "G1_PELVIS_F_4123" in source
    assert "/home/unitree/miniforge3/envs/egonav-deploy/bin/python" in source
    assert "9fd82e8a530408897fdd47c7c1b2f4314778a91339fcedd5db07a03b8c2b1d20" in source
    assert "--expected-robot-machine-id-sha256" in source
    assert "--rigid-body-id 42" in source
    assert "634e7de6eb7023b67beef8048c9b6d0aa26752641b40fe6adab34c8ce329e8cf" in source
    assert "f2e9c375304a3ceb0dd8998d58cda861b529ee8b4668e64c5dd7cf0d6cae2fc4" in source
    assert "3ce3af22d5eaf49d46878b8a9a07c86a7b0a132b6ba0c7ccb95487d7dc25863c" in source
    assert "bce9bf243b9ba864e8c5a9ab41b5b18a6f45fe9fd0c835a37ec7cdcbd85c334c" in source
    assert "--motive-map-transform" in source
    assert "--without-motive-ground-truth" in source
    assert "--record-unmapped-motive" in source
    assert "rt/lowcmd" not in source
    assert "run_g1_amo_operator.sh" not in source


def test_live_verification_validator_enforces_g1_4123_admission() -> None:
    source = (ROOT / "scripts/validate_g1_kiss_live_verification.py").read_text()
    assert '== "G1_PELVIS_F_4123"' in source
    assert "== 42" in source
    assert ">= 0.99" in source
    assert "availability >= 0.95" in source
    assert 'without external ground truth' in source
    assert '"absolute_map_pose_accuracy_evaluated": False' in source
    assert '"hardware_actuation_clearance": False' in source


def test_offline_qualification_uses_both_frozen_stress_runs() -> None:
    source = (
        ROOT / "scripts/run_g1_kiss_live_localization_offline_qualification.sh"
    ).read_text()
    assert "walk02" in source
    assert "walk03" in source
    assert "motive_front_pelvis_evaluator_only" in source
    assert "false_healthy_count" in source
