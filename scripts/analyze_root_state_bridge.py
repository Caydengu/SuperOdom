#!/usr/bin/env python3
"""Apply the strict runtime gate to a root-state replay JSONL artifact."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys

from g1_root_state_bridge.replay_analysis import (
    ReplayGateConfig,
    analyze_packet_trace,
)


def _timing(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "count": len(values),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--allowed-calibration-sha256", action="append", default=[])
    parser.add_argument("--max-state-age-ms", type=float, default=10.0)
    parser.add_argument("--max-correction-age-ms", type=float, default=250.0)
    parser.add_argument("--trace-end-time-ns", type=int)
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    metadata: dict[str, object] | None = None
    with args.input_jsonl.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                record = {
                    "kind": "packet",
                    "receipt_time_ns": 0,
                    "payload_hex": "not-hex",
                    "json_error": f"line {line_number}: {error}",
                }
            if record.get("kind") == "metadata" and metadata is None:
                metadata = record
            records.append(record)

    digest_hexes = list(args.allowed_calibration_sha256)
    if not digest_hexes and metadata is not None:
        candidate = metadata.get("calibration_digest_sha256")
        if isinstance(candidate, str):
            digest_hexes.append(candidate)
    digests: set[bytes] = set()
    for value in digest_hexes:
        try:
            digest = bytes.fromhex(value)
        except ValueError as error:
            parser.error(f"invalid calibration SHA-256 {value!r}: {error}")
        if len(digest) != 32:
            parser.error("allowed calibration SHA-256 must encode exactly 32 bytes")
        digests.add(digest)
    if not digests:
        parser.error("an allowlisted calibration digest is required")

    packet_receipts = [
        int(record["receipt_time_ns"])
        for record in records
        if record.get("kind") == "packet" and "receipt_time_ns" in record
    ]
    if args.trace_end_time_ns is not None:
        trace_end_time_ns = args.trace_end_time_ns
    elif packet_receipts:
        trace_end_time_ns = max(packet_receipts)
    else:
        trace_end_time_ns = 0

    report = analyze_packet_trace(
        records,
        config=ReplayGateConfig(
            max_state_age_ns=int(args.max_state_age_ms * 1_000_000),
            max_correction_age_ns=int(args.max_correction_age_ms * 1_000_000),
            allowed_calibration_digests=frozenset(digests),
        ),
        trace_end_time_ns=trace_end_time_ns,
    )
    output = asdict(report)
    output["state_age_summary_ns"] = _timing(report.state_age_ns)
    output["correction_age_summary_ns"] = _timing(report.correction_age_ns)
    output["joint_sync_gap_summary_ns"] = _timing(report.joint_sync_gap_ns)
    del output["state_age_ns"]
    del output["correction_age_ns"]
    del output["joint_sync_gap_ns"]
    output["trace_end_time_ns"] = trace_end_time_ns
    output["strict_gate_pass"] = (
        report.total_packets > 0
        and report.rejected_packets == 0
        and not report.publisher_loss
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite analysis artifact: {args.output_json}")
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0 if output["strict_gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
