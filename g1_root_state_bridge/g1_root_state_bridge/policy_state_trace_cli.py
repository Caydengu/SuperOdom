"""CLI for deterministic bridge-status to policy-state replay conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from g1_root_state_bridge.policy_state_trace import (
    PolicyStateTraceError,
    write_policy_state_replay,
    write_simulation_pair,
)


def _digest(value: str) -> bytes:
    try:
        digest = bytes.fromhex(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "digest must be 64 hexadecimal characters"
        ) from error
    if len(digest) != 32:
        raise argparse.ArgumentTypeError(
            "digest must be 64 hexadecimal characters"
        )
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a strict 50 Hz atomic G1 policy-state replay window "
            "from retained bridge evidence."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bridge-status-jsonl", type=Path)
    source.add_argument("--write-simulation-pair", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--samples", type=int, default=522)
    parser.add_argument("--simulation-samples", type=int, default=2)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--warmup-seconds", type=float, default=30.0)
    parser.add_argument(
        "--expected-calibration-sha256",
        type=_digest,
        required=True,
    )
    parser.add_argument(
        "--expected-joint-mapping-sha256",
        type=_digest,
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.write_simulation_pair is not None:
        try:
            write_simulation_pair(
                args.write_simulation_pair,
                calibration_digest=args.expected_calibration_sha256,
                joint_mapping_digest=args.expected_joint_mapping_sha256,
                samples=args.simulation_samples,
            )
        except (OSError, PolicyStateTraceError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 1
        return 0

    if args.output_jsonl is None or args.summary_json is None:
        print(
            "--bridge-status-jsonl requires --output-jsonl and "
            "--summary-json",
            file=sys.stderr,
        )
        return 2
    assert args.bridge_status_jsonl is not None
    if args.summary_json.exists():
        print(
            f"refusing to overwrite summary: {args.summary_json}",
            file=sys.stderr,
        )
        return 2
    try:
        report = write_policy_state_replay(
            args.bridge_status_jsonl,
            args.output_jsonl,
            samples=args.samples,
            rate_hz=args.rate_hz,
            warmup_seconds=args.warmup_seconds,
            expected_calibration_digest=args.expected_calibration_sha256,
            expected_joint_mapping_digest=args.expected_joint_mapping_sha256,
        )
    except (OSError, PolicyStateTraceError, ValueError) as error:
        payload: dict[str, object] = {
            "schema": "g1_policy_state_replay_summary_v1",
            "valid": False,
            "error": str(error),
        }
        if isinstance(error, PolicyStateTraceError) and error.report is not None:
            payload["report"] = error.report.to_json_dict()
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(str(error), file=sys.stderr)
        return 1

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(
            {
                "schema": "g1_policy_state_replay_summary_v1",
                "valid": True,
                "report": report.to_json_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
