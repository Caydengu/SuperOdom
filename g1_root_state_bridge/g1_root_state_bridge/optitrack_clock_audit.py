"""Audit the NatNet-to-Oslo clock mapping captured in raw reference JSONL."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from g1_root_state_bridge.dynamic_reference import ClockMapping


@dataclass(frozen=True)
class OptiTrackClockAuditReport:
    valid: bool
    gates_pass: bool
    invalid_reasons: tuple[str, ...]
    metrics: Mapping[str, float | int]
    gate_results: Mapping[str, bool]
    clock_mapping: ClockMapping | None
    audit_scope: str = (
        "NatNet Cristian clock-mapping consistency and transport timing; "
        "not independent PTP absolute-time certification"
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _strictly_increasing(values: Sequence[int | float]) -> bool:
    return all(previous < current for previous, current in zip(values, values[1:]))


def audit_optitrack_clock_records(
    records: Sequence[Mapping[str, object]],
    *,
    maximum_residual_p95_ns: int,
    minimum_frames: int,
    calibration_id: str,
) -> OptiTrackClockAuditReport:
    if maximum_residual_p95_ns < 0:
        raise ValueError("maximum_residual_p95_ns must be non-negative")
    if minimum_frames < 3:
        raise ValueError("minimum_frames must be at least three")
    if not calibration_id:
        raise ValueError("calibration_id must be non-empty")

    raw_frames = [
        record for record in records if record.get("kind") == "raw_rigid_body_pose"
    ]
    usable = [
        record
        for record in raw_frames
        if isinstance(record.get("motive_software_time_s"), (int, float))
        and math.isfinite(float(record["motive_software_time_s"]))
        and isinstance(record.get("capture_realtime_estimate_ns"), int)
    ]
    metrics: dict[str, float | int] = {
        "raw_frame_count": len(raw_frames),
        "usable_frame_count": len(usable),
        "clock_frame_fraction": (
            len(usable) / len(raw_frames) if raw_frames else 0.0
        ),
    }
    reasons: list[str] = []
    if len(usable) < minimum_frames:
        reasons.append("insufficient_clock_frames")
        return OptiTrackClockAuditReport(
            valid=False,
            gates_pass=False,
            invalid_reasons=tuple(reasons),
            metrics=metrics,
            gate_results={},
            clock_mapping=None,
        )

    try:
        frame_numbers = [int(record["frame_number"]) for record in usable]
        source_seconds = [
            float(record["motive_software_time_s"]) for record in usable
        ]
        capture_times_ns = [
            int(record["capture_realtime_estimate_ns"]) for record in usable
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("usable clock record is malformed") from exc

    if not _strictly_increasing(frame_numbers):
        reasons.append("frame_number_not_strictly_increasing")
    if not _strictly_increasing(source_seconds):
        reasons.append("motive_time_not_strictly_increasing")
    if not _strictly_increasing(capture_times_ns):
        reasons.append("capture_time_not_strictly_increasing")

    frame_differences = np.diff(np.asarray(frame_numbers, dtype=np.int64))
    metrics["frame_number_gap_count"] = int(np.sum(frame_differences > 1))
    metrics["estimated_dropped_frame_count"] = int(
        np.sum(np.maximum(frame_differences - 1, 0))
    )
    if reasons:
        return OptiTrackClockAuditReport(
            valid=False,
            gates_pass=False,
            invalid_reasons=tuple(reasons),
            metrics=metrics,
            gate_results={},
            clock_mapping=None,
        )

    source_ns = np.asarray(source_seconds, dtype=np.float64) * 1.0e9
    target_ns = np.asarray(capture_times_ns, dtype=np.float64)
    source_centered = source_ns - np.mean(source_ns)
    target_centered = target_ns - np.mean(target_ns)
    denominator = float(np.dot(source_centered, source_centered))
    if denominator <= 0.0:
        return OptiTrackClockAuditReport(
            valid=False,
            gates_pass=False,
            invalid_reasons=("clock_fit_is_degenerate",),
            metrics=metrics,
            gate_results={},
            clock_mapping=None,
        )
    scale = float(np.dot(source_centered, target_centered) / denominator)
    offset_ns = float(np.mean(target_ns) - scale * np.mean(source_ns))
    predicted_ns = scale * source_ns + offset_ns
    absolute_residuals_ns = np.abs(target_ns - predicted_ns)
    mapping_residual_p95_ns = _percentile(absolute_residuals_ns, 95.0)
    mapping_residual_max_ns = float(np.max(absolute_residuals_ns))

    client_latency_ms = [
        float(record["seconds_since_host_mid_exposure"]) * 1000.0
        for record in usable
        if isinstance(
            record.get("seconds_since_host_mid_exposure"),
            (int, float),
        )
        and math.isfinite(float(record["seconds_since_host_mid_exposure"]))
    ]
    system_latency_ms: list[float] = []
    software_latency_ms: list[float] = []
    for record in usable:
        try:
            frequency = int(record["high_resolution_clock_frequency_hz"])
            mid = int(record["camera_mid_exposure_ticks"])
            received = int(record["camera_data_received_ticks"])
            transmitted = int(record["transmit_ticks"])
        except (KeyError, TypeError, ValueError):
            continue
        if frequency <= 0 or not (mid <= received <= transmitted):
            continue
        system_latency_ms.append((transmitted - mid) * 1000.0 / frequency)
        software_latency_ms.append(
            (transmitted - received) * 1000.0 / frequency
        )

    metrics.update(
        {
            "mapping_scale": scale,
            "mapping_offset_ns": int(round(offset_ns)),
            "mapping_scale_drift_ppm": (scale - 1.0) * 1.0e6,
            "mapping_residual_p50_ns": _percentile(
                absolute_residuals_ns, 50.0
            ),
            "mapping_residual_p95_ns": mapping_residual_p95_ns,
            "mapping_residual_max_ns": mapping_residual_max_ns,
            "source_duration_s": source_seconds[-1] - source_seconds[0],
            "client_latency_count": len(client_latency_ms),
            "system_latency_count": len(system_latency_ms),
        }
    )
    if client_latency_ms:
        metrics.update(
            {
                "client_latency_p50_ms": _percentile(
                    client_latency_ms, 50.0
                ),
                "client_latency_p95_ms": _percentile(
                    client_latency_ms, 95.0
                ),
                "client_latency_max_ms": max(client_latency_ms),
            }
        )
    if system_latency_ms:
        metrics.update(
            {
                "system_latency_p50_ms": _percentile(
                    system_latency_ms, 50.0
                ),
                "system_latency_p95_ms": _percentile(
                    system_latency_ms, 95.0
                ),
                "system_latency_max_ms": max(system_latency_ms),
                "software_latency_p95_ms": _percentile(
                    software_latency_ms, 95.0
                ),
            }
        )

    gate_results = {
        "mapping_residual_p95": (
            mapping_residual_p95_ns <= maximum_residual_p95_ns
        )
    }
    mapping = ClockMapping(
        source_clock_id="motive_software_ns",
        target_clock_id="unix_realtime_ns",
        scale=scale,
        offset_ns=int(round(offset_ns)),
        residual_p95_ns=int(math.ceil(mapping_residual_p95_ns)),
        calibration_id=calibration_id,
    )
    return OptiTrackClockAuditReport(
        valid=True,
        gates_pass=all(gate_results.values()),
        invalid_reasons=(),
        metrics=metrics,
        gate_results=gate_results,
        clock_mapping=mapping,
    )

