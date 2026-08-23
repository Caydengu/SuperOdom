from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_fast_lio_replay_uses_seconds_and_frozen_g1_topics() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "g1_root_state_bridge/g1_root_state_bridge/config"
            / "fast_lio_g1_4123_replay_seconds.yaml"
        ).read_text(encoding="utf-8")
    )["/**"]["ros__parameters"]
    assert config["common"]["lid_topic"] == "/utlidar/cloud_livox_mid360"
    assert config["common"]["imu_topic"] == "/utlidar/imu_livox_mid360"
    assert config["preprocess"]["timestamp_unit"] == 0
    assert config["mapping"]["extrinsic_est_en"] is False
    assert config["mapping"]["extrinsic_R"] == [
        1.0,
        0.0,
        0.0,
        0.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        -1.0,
    ]


def test_fast_lio_replay_is_output_only() -> None:
    script = (ROOT / "scripts/replay_fast_lio_dataset.sh").read_text(
        encoding="utf-8"
    )
    assert "ros2 bag record --use-sim-time" in script
    assert 'source "$install/setup.bash"' in script
    assert "/Odometry" in script
    for forbidden in ("sportmodestate", "rt/lowcmd", "LowCmd"):
        assert forbidden not in script
