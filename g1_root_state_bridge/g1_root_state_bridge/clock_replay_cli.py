"""Replay captured LowState clock pairs through the live clock admission path."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from g1_root_state_bridge.clock_sync import (
    ClockMapConfig,
    OnlineAffineClockMapper,
    fit_affine_lower_envelope_clock,
)
from g1_root_state_bridge.dynamic_capture_io import iter_recorded_datagrams


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def analyze_clock_observations(
    source_time_ns: np.ndarray,
    receipt_time_ns: np.ndarray,
    *,
    config: ClockMapConfig = ClockMapConfig(),
    minimum_valid_fraction: float = 0.995,
    maximum_batch_delta_p95_ms: float = 0.5,
    maximum_observe_p95_us: float = 100.0,
) -> dict[str, object]:
    """Compare causal online mapping to the batch lower-envelope oracle."""

    source = np.asarray(source_time_ns, dtype=np.int64)
    receipt = np.asarray(receipt_time_ns, dtype=np.int64)
    if source.ndim != 1 or receipt.shape != source.shape or source.size < 2:
        raise ValueError("clock replay needs equal one-dimensional timestamp arrays")
    batch = fit_affine_lower_envelope_clock(source, receipt, lower_quantile=0.01)
    mapper = OnlineAffineClockMapper(config)
    mapped: list[int] = []
    epochs: list[int] = []
    fit_valid: list[bool] = []
    sample_valid: list[bool] = []
    reasons: Counter[str] = Counter()
    runtime_us: list[float] = []
    for source_ns, receipt_ns in zip(source, receipt, strict=True):
        started_ns = time.perf_counter_ns()
        estimate = mapper.observe(int(source_ns), int(receipt_ns))
        runtime_us.append((time.perf_counter_ns() - started_ns) * 1e-3)
        mapped.append(estimate.mapped_time_ns)
        epochs.append(estimate.source_epoch)
        fit_valid.append(estimate.fit_valid)
        sample_valid.append(estimate.sample_valid)
        reasons.update(estimate.health_reasons)

    warm = (source - source[0] >= config.minimum_span_ns) & (
        np.arange(source.size) >= config.minimum_samples - 1
    )
    if not np.any(warm):
        raise ValueError("capture is shorter than the clock admission span")
    mapped_array = np.asarray(mapped, dtype=np.int64)
    batch_mapped = np.asarray([batch.map_time_ns(int(value)) for value in source], dtype=np.int64)
    delta_ms = np.abs(mapped_array[warm] - batch_mapped[warm]) * 1e-6
    fit_valid_array = np.asarray(fit_valid, dtype=np.bool_)[warm]
    sample_valid_array = np.asarray(sample_valid, dtype=np.bool_)[warm]
    runtime_array = np.asarray(runtime_us, dtype=np.float64)[warm]
    epoch_count = len(set(epochs))
    fit_fraction = float(np.mean(fit_valid_array))
    sample_fraction = float(np.mean(sample_valid_array))
    delta = _quantiles(delta_ms)
    runtime = _quantiles(runtime_array)
    mapped_monotonic = bool(np.all(np.diff(mapped_array) > 0))
    passed = (
        epoch_count == 1
        and mapped_monotonic
        and fit_fraction == 1.0
        and sample_fraction >= minimum_valid_fraction
        and delta["p95"] <= maximum_batch_delta_p95_ms
        and runtime["p95"] <= maximum_observe_p95_us
    )
    return {
        "schema": "g1_online_clock_replay_v1",
        "status": "pass" if passed else "fail",
        "samples": int(source.size),
        "samples_after_warmup": int(np.count_nonzero(warm)),
        "epoch_count": epoch_count,
        "mapped_monotonic": mapped_monotonic,
        "fit_valid_fraction": fit_fraction,
        "sample_valid_fraction": sample_fraction,
        "health_reason_counts": dict(sorted(reasons.items())),
        "online_to_batch_abs_delta_ms": delta,
        "observe_runtime_us": runtime,
        "batch_slope": batch.slope,
        "online_final_slope": mapper.estimate(int(source[-1])).slope,
        "thresholds": {
            "minimum_valid_fraction": minimum_valid_fraction,
            "maximum_batch_delta_p95_ms": maximum_batch_delta_p95_ms,
            "maximum_observe_p95_us": maximum_observe_p95_us,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lowstate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-valid-fraction", type=float, default=0.995)
    parser.add_argument("--maximum-batch-delta-p95-ms", type=float, default=0.5)
    parser.add_argument("--maximum-observe-p95-us", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source: list[int] = []
    receipt: list[int] = []
    for recorded in iter_recorded_datagrams(args.lowstate):
        source.append(int(recorded.packet.robot_stamp_ns))
        receipt.append(int(recorded.receipt_realtime_ns))
    report = analyze_clock_observations(
        np.asarray(source, dtype=np.int64),
        np.asarray(receipt, dtype=np.int64),
        minimum_valid_fraction=args.minimum_valid_fraction,
        maximum_batch_delta_p95_ms=args.maximum_batch_delta_p95_ms,
        maximum_observe_p95_us=args.maximum_observe_p95_us,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
