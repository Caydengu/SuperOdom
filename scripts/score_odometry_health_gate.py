#!/usr/bin/env python3
"""Replay the fail-closed local-odometry health gate on exported tracks."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from g1_root_state_bridge.odometry_health import (
    OdometryHealthConfig,
    OdometryHealthGate,
)


def _wxyz(xyzw: list[float]) -> tuple[float, float, float, float]:
    x, y, z, w = xyzw
    return (w, x, y, z)


def score(
    path: Path,
    *,
    topic: str,
    config: OdometryHealthConfig,
    reference_event_time_ns: int | None,
) -> dict[str, Any]:
    gate = OdometryHealthGate(config)
    reasons: Counter[str] = Counter()
    first_violation_time_ns: int | None = None
    first_healthy_time_ns: int | None = None
    last_time_ns: int | None = None
    sample_count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("kind") != "odometry" or row.get("topic") != topic:
            continue
        state = gate.update(
            source_time_ns=int(row["source_time_ns"]),
            position_xyz_m=row["position_xyz_m"],
            quaternion_wxyz=_wxyz(row["quaternion_xyzw"]),
        )
        sample_count += 1
        last_time_ns = state.source_time_ns
        reasons[state.reason] += 1
        if state.healthy and first_healthy_time_ns is None:
            first_healthy_time_ns = state.source_time_ns
        if state.reason in {"planar_speed", "yaw_rate", "planar_speed+yaw_rate"}:
            if first_violation_time_ns is None:
                first_violation_time_ns = state.source_time_ns

    if sample_count == 0:
        raise ValueError(f"no odometry rows found for topic {topic!r}")
    evaluated_count = sample_count - reasons["warming_up"]
    violation_count = sum(
        reasons[reason]
        for reason in ("planar_speed", "yaw_rate", "planar_speed+yaw_rate")
    )
    lead_s = None
    if reference_event_time_ns is not None and first_violation_time_ns is not None:
        lead_s = (reference_event_time_ns - first_violation_time_ns) * 1e-9
    return {
        "schema": "g1_odometry_health_gate_replay_v1",
        "input": str(path),
        "topic": topic,
        "config": {
            "window_ns": config.window_ns,
            "max_planar_speed_mps": config.max_planar_speed_mps,
            "max_yaw_rate_radps": config.max_yaw_rate_radps,
            "recovery_hold_ns": config.recovery_hold_ns,
        },
        "sample_count": sample_count,
        "evaluated_count_after_warmup": evaluated_count,
        "reason_counts": dict(sorted(reasons.items())),
        "violation_count": violation_count,
        "violation_fraction_after_warmup": (
            violation_count / evaluated_count if evaluated_count else None
        ),
        "first_healthy_time_ns": first_healthy_time_ns,
        "first_violation_time_ns": first_violation_time_ns,
        "last_time_ns": last_time_ns,
        "reference_event_time_ns": reference_event_time_ns,
        "first_violation_lead_s": lead_s,
        "interpretation_boundary": (
            "trajectory plausibility only; no map consistency, covariance, command, "
            "or external ground-truth evidence is used by the gate"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", default="/pelvis_state_estimation")
    parser.add_argument("--window-ms", type=float, default=200.0)
    parser.add_argument("--max-planar-speed-mps", type=float, default=1.5)
    parser.add_argument("--max-yaw-rate-radps", type=float, default=2.0)
    parser.add_argument("--recovery-ms", type=float, default=1000.0)
    parser.add_argument("--reference-event-time-ns", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output}")
    config = OdometryHealthConfig(
        window_ns=round(args.window_ms * 1e6),
        max_planar_speed_mps=args.max_planar_speed_mps,
        max_yaw_rate_radps=args.max_yaw_rate_radps,
        recovery_hold_ns=round(args.recovery_ms * 1e6),
    )
    result = score(
        args.tracks,
        topic=args.topic,
        config=config,
        reference_event_time_ns=args.reference_event_time_ns,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
