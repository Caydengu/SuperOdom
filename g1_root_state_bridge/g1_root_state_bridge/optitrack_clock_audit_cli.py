"""Command-line wrapper for the raw NatNet clock-consistency audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from g1_root_state_bridge.optitrack_clock_audit import (
    audit_optitrack_clock_records,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit NatNet capture-time mapping consistency."
    )
    parser.add_argument("--raw-jsonl", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--calibration-id", required=True)
    parser.add_argument(
        "--maximum-residual-p95-ms",
        type=float,
        default=5.0,
    )
    parser.add_argument("--minimum-frames", type=int, default=300)
    args = parser.parse_args(argv)

    if not args.raw_jsonl.is_file():
        parser.error(f"input does not exist: {args.raw_jsonl}")
    if args.output_json.exists():
        parser.error(f"refusing to overwrite output: {args.output_json}")
    try:
        report = audit_optitrack_clock_records(
            _load_records(args.raw_jsonl),
            maximum_residual_p95_ns=int(
                round(args.maximum_residual_p95_ms * 1.0e6)
            ),
            minimum_frames=args.minimum_frames,
            calibration_id=args.calibration_id,
        )
    except ValueError as exc:
        parser.error(str(exc))

    mapping = None
    if report.clock_mapping is not None:
        mapping = {
            "source_clock_id": report.clock_mapping.source_clock_id,
            "target_clock_id": report.clock_mapping.target_clock_id,
            "scale": report.clock_mapping.scale,
            "offset_ns": report.clock_mapping.offset_ns,
            "residual_p95_ns": report.clock_mapping.residual_p95_ns,
            "calibration_id": report.clock_mapping.calibration_id,
        }
    payload = {
        "schema": "g1_optitrack_clock_audit_v1",
        "valid": report.valid,
        "gates_pass": report.gates_pass,
        "invalid_reasons": list(report.invalid_reasons),
        "metrics": dict(report.metrics),
        "gate_results": dict(report.gate_results),
        "clock_mapping": mapping,
        "audit_scope": report.audit_scope,
        "raw_jsonl": str(args.raw_jsonl.resolve()),
        "raw_sha256": _sha256(args.raw_jsonl),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not report.valid:
        return 3
    return 0 if report.gates_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

