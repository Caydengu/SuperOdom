#!/usr/bin/env python3
"""Score recorded joint and LiDAR outages against the fail-closed contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from g1_root_state_bridge.fault_analysis import analyze_source_faults


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--fault-events-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--joint-freshness-ms", type=float, default=10.0)
    parser.add_argument("--lidar-freshness-ms", type=float, default=250.0)
    args = parser.parse_args()

    if args.joint_freshness_ms <= 0 or args.lidar_freshness_ms <= 0:
        parser.error("freshness limits must be positive")
    records: list[dict[str, object]] = []
    with args.input_jsonl.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(record)
    fault_events = json.loads(args.fault_events_json.read_text(encoding="utf-8"))
    if not isinstance(fault_events, dict):
        raise ValueError("fault-events artifact must contain an object")

    report = analyze_source_faults(
        records,
        fault_events=fault_events,
        freshness_ns={
            "joint_source": int(args.joint_freshness_ms * 1_000_000),
            "lidar_source": int(args.lidar_freshness_ms * 1_000_000),
        },
    )
    report["input_jsonl"] = str(args.input_jsonl)
    report["fault_events_json"] = str(args.fault_events_json)
    report["freshness_limits_ms"] = {
        "joint_source": args.joint_freshness_ms,
        "lidar_source": args.lidar_freshness_ms,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.output_json.exists():
        raise FileExistsError(
            f"refusing to overwrite analysis artifact: {args.output_json}"
        )
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["all_fault_windows_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
