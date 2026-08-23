#!/usr/bin/env python3
"""Generate matched root-aware treatments from one SuperOdometry sensor track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from g1_root_state_bridge.amo_dataset import load_lowstate
from g1_root_state_bridge.amo_treatments import generate_treatments, load_odometry_track


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--odometry-track", type=Path, required=True)
    parser.add_argument("--lowstate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--odometry-time-domain",
        choices=("robot", "oslo_event"),
        default="robot",
        help=(
            "Clock domain of odometry source_time_ns. Use oslo_event only "
            "after a documented sensor-header-to-Oslo clock mapping."
        ),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    loaded = load_odometry_track(args.odometry_track)
    source = loaded.records
    lowstate = load_lowstate(args.lowstate)
    counts: dict[str, int] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "schema": "g1_amo_localization_treatment_track_v1",
                    "kind": "metadata",
                    "odometry_track": str(args.odometry_track),
                    "lowstate": str(args.lowstate),
                    "motive_online_input": False,
                    "physical_sensor_T_observed": "identity",
                    "odometry_time_domain": args.odometry_time_domain,
                    "source_odometry_health": loaded.health,
                },
                sort_keys=True,
            )
            + "\n"
        )
        for record in generate_treatments(
            source,
            lowstate,
            source_time_domain=args.odometry_time_domain,
        ):
            treatment = str(record["treatment"])
            counts[treatment] = counts.get(treatment, 0) + 1
            stream.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_odometry_records": loaded.total_records,
                "source_odometry_health": loaded.health,
                "counts": counts,
                "skipped_source_records": loaded.total_records
                - counts.get("superodom_sensor", 0),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
