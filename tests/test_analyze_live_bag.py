from __future__ import annotations

import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/analyze_live_bag.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_live_bag", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_series_metrics_report_rate_age_and_ordering() -> None:
    module = load_module()
    result = module.series_metrics(
        header_ns=[1_000_000_000, 1_100_000_000, 1_200_000_000],
        receipt_ns=[1_010_000_000, 1_115_000_000, 1_225_000_000],
    )

    assert result["count"] == 3
    assert result["header_nonmonotonic_count"] == 0
    assert result["receipt_nonmonotonic_count"] == 0
    assert math.isclose(result["header_rate_hz"], 10.0)
    assert math.isclose(result["receipt_age_ms"]["p50"], 15.0)
    assert math.isclose(result["receipt_age_ms"]["p95"], 24.0)
    assert math.isclose(result["receipt_age_ms"]["maximum"], 25.0)
    assert math.isclose(result["receipt_age_first_ms"], 10.0)
    assert math.isclose(result["receipt_age_last_ms"], 25.0)
    assert math.isclose(result["receipt_age_growth_ms_per_s"], 75.0)
    assert math.isclose(result["header_interarrival_ms"]["p50"], 100.0)

    reordered = module.series_metrics(
        header_ns=[10, 10, 9],
        receipt_ns=[20, 21, 19],
    )
    assert reordered["header_nonmonotonic_count"] == 2
    assert reordered["receipt_nonmonotonic_count"] == 1


def test_trajectory_metrics_report_stationary_drift_yaw_and_jumps() -> None:
    module = load_module()
    yaws = [0.0, 0.1, 0.2]
    quaternions = [
        (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
        for yaw in yaws
    ]
    result = module.trajectory_metrics(
        positions=[(0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.03, 0.04, 0.0)],
        quaternions=quaternions,
        timestamps_ns=[1_000_000_000, 1_100_000_000, 1_200_000_000],
        warmup_s=0.1,
    )

    assert math.isclose(result["terminal_translation_from_start_m"], 0.05)
    assert math.isclose(result["maximum_translation_from_start_m"], 0.05)
    assert math.isclose(result["maximum_consecutive_translation_m"], math.sqrt(0.002))
    assert math.isclose(result["terminal_abs_yaw_from_start_deg"], math.degrees(0.2))
    assert result["nonfinite_sample_count"] == 0
    assert result["maximum_quaternion_norm_error"] < 1e-12
    assert math.isclose(result["maximum_consecutive_translation_at_elapsed_s"], 0.2)
    assert math.isclose(
        result["post_warmup"]["terminal_translation_from_warmup_start_m"],
        math.sqrt(0.002),
    )
    assert math.isclose(
        result["post_warmup"]["terminal_abs_yaw_from_warmup_start_deg"],
        math.degrees(0.1),
    )


def test_timestamp_matching_counts_exact_output_to_imu_correspondence() -> None:
    module = load_module()
    result = module.timestamp_correspondence(
        source_ns=[100, 200, 300, 400],
        output_ns=[200, 300, 350],
    )

    assert result == {
        "source_unique_count": 4,
        "output_unique_count": 3,
        "output_matched_count": 2,
        "output_unmatched_count": 1,
        "output_match_fraction": 2 / 3,
    }


def test_optimization_stats_metrics_preserve_cadence_and_processing_cost() -> None:
    module = load_module()
    result = module.optimization_stats_metrics(
        header_ns=[1_000_000_000, 1_100_000_000, 1_200_000_000],
        receipt_ns=[1_010_000_000, 1_111_000_000, 1_212_000_000],
        optimization_ms=[8.0, 10.0, 12.0],
        frame_processing_ms=[11.0, 13.0, 15.0],
    )

    assert result["count"] == 3
    assert math.isclose(result["header_rate_hz"], 10.0)
    assert math.isclose(result["optimization_ms"]["p95"], 11.8)
    assert math.isclose(result["frame_processing_ms"]["maximum"], 15.0)


def test_correction_timing_metrics_keep_pose_reference_and_evidence_separate() -> None:
    module = load_module()
    result = module.correction_timing_metrics(
        reference_ns=[1_000_000_000, 1_100_000_000],
        evidence_ns=[1_099_000_000, 1_199_500_000],
        mapping_output_ns=[1_125_000_000, 1_226_000_000],
        application_ns=[1_127_000_000, 1_228_500_000],
        receipt_ns=[1_128_000_000, 1_229_000_000],
        sequences=[1, 2],
        reset_ids=[7, 7],
        valid=[True, True],
    )

    assert result["count"] == 2
    assert result["valid_count"] == 2
    assert result["timing_order_violation_count"] == 0
    assert result["sequence_nonmonotonic_count"] == 0
    assert math.isclose(result["scan_duration_ms"]["p50"], 99.25)
    assert math.isclose(result["correction_evidence_gap_ms"]["p50"], 100.5)
    assert math.isclose(result["mapping_delay_from_newest_observation_ms"]["p50"], 26.25)
    assert math.isclose(result["estimator_application_delay_ms"]["p50"], 2.25)
    assert math.isclose(result["evidence_age_at_bridge_receipt_ms"]["p50"], 29.25)
