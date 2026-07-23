"""Normalize raw NatNet records with an explicit direct-pelvis calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from g1_root_state_bridge.optitrack_reference import (
    OptiTrackReferenceError,
    convert_raw_optitrack_records,
    load_pelvis_calibration,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OptiTrackReferenceError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(record, dict):
                raise OptiTrackReferenceError(
                    f"{path}:{line_number}: record must be an object"
                )
            records.append(record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Motive-native rigid-body JSONL into the normalized "
            "direct-pelvis reference contract."
        )
    )
    parser.add_argument("--raw-jsonl", required=True, type=Path)
    parser.add_argument("--calibration-json", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    args = parser.parse_args(argv)

    for path in (args.raw_jsonl, args.calibration_json):
        if not path.is_file():
            parser.error(f"input does not exist: {path}")
    if args.output_jsonl.exists():
        parser.error(f"refusing to overwrite output: {args.output_jsonl}")

    try:
        calibration = load_pelvis_calibration(args.calibration_json)
        calibration_sha256 = _sha256(args.calibration_json)
        raw_sha256 = _sha256(args.raw_jsonl)
        normalized = convert_raw_optitrack_records(
            _load_jsonl(args.raw_jsonl),
            calibration=calibration,
            calibration_sha256=calibration_sha256,
        )
    except OptiTrackReferenceError as exc:
        parser.error(str(exc))

    normalized[0]["raw_capture_sha256"] = raw_sha256
    normalized[0]["raw_capture_path"] = str(args.raw_jsonl.resolve())
    normalized[0]["calibration_path"] = str(args.calibration_json.resolve())
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("x", encoding="utf-8") as stream:
        for record in normalized:
            stream.write(
                json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
            )

    valid_count = sum(
        record.get("kind") == "reference_pose"
        and record.get("tracking_valid") is True
        for record in normalized
    )
    total_count = sum(
        record.get("kind") == "reference_pose" for record in normalized
    )
    print(
        json.dumps(
            {
                "total_reference_poses": total_count,
                "valid_reference_poses": valid_count,
                "output_jsonl": str(args.output_jsonl.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0 if valid_count > 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())

