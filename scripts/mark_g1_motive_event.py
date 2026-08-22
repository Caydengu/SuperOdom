#!/usr/bin/env python3
"""Append one host-clock operator event to a G1/Motive capture."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import re
import time


LABEL_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    if not args.run_dir.is_dir() or not (args.run_dir / "manifest.json").is_file():
        raise SystemExit("--run-dir must be an active G1/Motive capture directory")
    if LABEL_PATTERN.fullmatch(args.label) is None:
        raise SystemExit("--label must match [a-z0-9][a-z0-9_-]{0,63}")

    events_dir = args.run_dir / "events"
    events_dir.mkdir(exist_ok=True)
    output = events_dir / "operator_events.jsonl"
    record = {
        "schema": "g1_motive_operator_event_v1",
        "label": args.label,
        "note": args.note,
        "realtime_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
    }
    with output.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
