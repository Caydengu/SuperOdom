"""Command-line entry point for immutable dynamic-reference scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from g1_root_state_bridge.dynamic_reference import score_dynamic_reference
from g1_root_state_bridge.dynamic_reference_io import (
    DynamicReferenceFormatError,
    load_estimator_status_jsonl,
    load_reference_pose_jsonl,
    load_score_config_json,
    load_waist_joint_evidence_jsonl,
    report_as_dict,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score typed SuperOdometry pelvis packets against a normalized "
            "independent pelvis reference."
        )
    )
    parser.add_argument("--estimator-jsonl", required=True, type=Path)
    parser.add_argument("--reference-jsonl", required=True, type=Path)
    parser.add_argument("--config-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--estimator-frame-id", default="superodom_world")
    args = parser.parse_args(argv)

    for path in (
        args.estimator_jsonl,
        args.reference_jsonl,
        args.config_json,
    ):
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
    if args.output_json.exists():
        parser.error(f"refusing to overwrite output: {args.output_json}")

    try:
        config, clock_mapping = load_score_config_json(args.config_json)
        estimator = load_estimator_status_jsonl(
            args.estimator_jsonl,
            clock_id=config.target_clock_id,
            frame_id=args.estimator_frame_id,
        )
        waist_joints = load_waist_joint_evidence_jsonl(
            args.estimator_jsonl
        )
        reference = load_reference_pose_jsonl(args.reference_jsonl)
        report = score_dynamic_reference(
            estimator,
            reference,
            config=config,
            clock_mapping=clock_mapping,
            waist_joint_samples=waist_joints,
        )
    except DynamicReferenceFormatError as exc:
        parser.error(str(exc))

    payload = report_as_dict(report)
    payload["inputs"] = {
        "estimator_jsonl": str(args.estimator_jsonl.resolve()),
        "estimator_sha256": _sha256(args.estimator_jsonl),
        "reference_jsonl": str(args.reference_jsonl.resolve()),
        "reference_sha256": _sha256(args.reference_jsonl),
        "config_json": str(args.config_json.resolve()),
        "config_sha256": _sha256(args.config_json),
        "estimator_frame_id": args.estimator_frame_id,
    }
    payload["sample_counts"] = {
        "estimator": len(estimator),
        "reference": len(reference),
        "waist_joint": len(waist_joints),
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
