#!/usr/bin/env python3
"""Audit G1 AMO/Motive captures and freeze evaluator-only walking boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from g1_root_state_bridge.amo_dataset import audit_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--boundaries-output", type=Path, required=True)
    parser.add_argument(
        "--visual-confirmed",
        action="store_true",
        help="Record that RGB contact sheets were reviewed once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audits: list[dict[str, object]] = []
    boundaries: dict[str, object] = {}
    for run_dir in args.run:
        audit, boundary = audit_run(run_dir)
        audits.append(audit)
        boundaries[run_dir.name] = {
            **boundary.as_dict(),
            "visual_confirmation": args.visual_confirmed,
            "role": "development"
            if run_dir.name.endswith("walk-01")
            else "frozen_primary_holdout",
        }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.boundaries_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(
            {"schema": "g1_motive_amo_dataset_audit_set_v1", "runs": audits}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    args.boundaries_output.write_text(
        json.dumps(
            {
                "schema": "g1_motive_amo_frozen_boundaries_v1",
                "ground_truth_policy": "evaluator_only",
                "method_specific_boundaries_forbidden": True,
                "runs": boundaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "audit_output": str(args.audit_output),
                "boundaries_output": str(args.boundaries_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
