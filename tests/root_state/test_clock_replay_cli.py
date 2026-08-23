import numpy as np

from g1_root_state_bridge.clock_replay_cli import analyze_clock_observations
from g1_root_state_bridge.clock_sync import ClockMapConfig


def test_clock_replay_admits_stable_clock_with_one_delayed_sample() -> None:
    source = 1_000_000_000 + np.arange(200, dtype=np.int64) * 30_000_000
    receipt = source + 2_000_000_000
    receipt[150] += 20_000_000
    report = analyze_clock_observations(
        source,
        receipt,
        config=ClockMapConfig(
            minimum_samples=8,
            minimum_span_ns=100_000_000,
            refit_period_samples=32,
        ),
        minimum_valid_fraction=0.99,
    )
    assert report["status"] == "pass"
    assert report["epoch_count"] == 1
    assert report["fit_valid_fraction"] == 1.0
    assert report["sample_valid_fraction"] < 1.0
    assert report["health_reason_counts"]["transport_delay"] >= 1
