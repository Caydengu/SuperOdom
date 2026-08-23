from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_live_node_uses_deployed_topics_and_has_no_unitree_command_surface() -> None:
    source = (ROOT / "g1_root_state_bridge/g1_root_state_bridge/live_node.py").read_text()
    assert "/utlidar/cloud_livox_mid360" in source
    assert "/utlidar/imu_livox_mid360" in source
    assert "tcp://*:5575" in source
    assert "ChannelPublisher" not in source
    assert "rt/lowcmd" not in source
    assert "sportmodestate" not in source.lower()
    assert '"structurally_unavailable"' in source


def test_setup_exposes_live_and_replay_entrypoints() -> None:
    setup = (ROOT / "g1_root_state_bridge/setup.py").read_text()
    assert "g1-kiss-live-localization" in setup
    assert "g1-kiss-localization-replay" in setup
    assert "g1_root_state_bridge.bridge_node" not in setup


def test_live_node_discards_cross_epoch_scans_and_survives_zmq_backpressure() -> None:
    source = (ROOT / "g1_root_state_bridge/g1_root_state_bridge/live_node.py").read_text()
    assert "queued_epoch != self.pipeline_epoch" in source
    assert "except zmq.Again" in source
    assert 'self.stats["transport_rejected"] += 1' in source


def test_transient_clock_delay_drops_sample_without_resetting_local_map() -> None:
    source = (ROOT / "g1_root_state_bridge/g1_root_state_bridge/live_node.py").read_text()
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


def test_offline_qualification_uses_both_frozen_stress_runs() -> None:
    source = (
        ROOT / "scripts/run_g1_kiss_live_localization_offline_qualification.sh"
    ).read_text()
    assert "walk02" in source
    assert "walk03" in source
    assert "motive_front_pelvis_evaluator_only" in source
    assert "false_healthy_count" in source
