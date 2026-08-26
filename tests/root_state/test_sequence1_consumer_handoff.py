from pathlib import Path


def test_live_launcher_subscribes_consumer_before_sequence1_publisher() -> None:
    source = Path("scripts/run_g1_kiss_live_verification.sh").read_text(
        encoding="utf-8"
    )
    subscription = source.index("--subscription-ready")
    map_publisher = source.index("run_g1_structural_map_shadow.sh")
    assert subscription < map_publisher
    assert "--wait-for-initialization-sec" in source
    assert "exact_sequence1_anchor" in source
    assert "ROBOT_VLM_MAP_POSITION_POLICY=initialization_only" in source
    assert '--tracking-period-sec "$live_duration"' in source
