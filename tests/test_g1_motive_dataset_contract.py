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
    assert "rt/lowcmd" not in source
    assert "ChannelPublisher" not in source
