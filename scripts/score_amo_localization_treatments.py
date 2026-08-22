#!/usr/bin/env python3
"""Score root-aware AMO localization treatments against evaluator-only Motive."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from g1_root_state_bridge.amo_scoring import score_treatments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--treatments", type=Path, required=True)
    parser.add_argument("--motive", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--front-plane-to-pelvis-x-m", type=float, default=0.0)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument(
        "--fixed-event-time-shift-ms",
        type=float,
        default=0.0,
        help="One development-frozen event association shift applied to every treatment.",
    )
    parser.add_argument(
        "--alignment-from-score",
        type=Path,
        help="Reuse the yaw-only world rotation from a development score JSON.",
    )
    parser.add_argument(
        "--score-window-evaluator-alignment",
        action="store_true",
        help=(
            "Fit one post-hoc yaw rotation from the sensor treatment and apply "
            "it to every treatment; measures local odometry, not deployable global heading."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.alignment_from_score is not None and args.score_window_evaluator_alignment:
        parser.error(
            "--alignment-from-score and --score-window-evaluator-alignment are mutually exclusive"
        )
    boundaries = json.loads(args.boundaries.read_text(encoding="utf-8"))
    boundary = int(boundaries["runs"][args.run_name]["event_realtime_ns"])
    rotation = None
    fixed_alignment_metadata = None
    if args.alignment_from_score is not None:
        development = json.loads(args.alignment_from_score.read_text(encoding="utf-8"))
        alignment = development["alignment"]
        matrix = alignment.get("rotation_matrix")
        if matrix is None:
            angle = math.radians(float(alignment["rotation_deg"]))
            matrix = [
                [math.cos(angle), math.sin(angle)],
                [-math.sin(angle), math.cos(angle)],
            ]
        rotation = matrix
        fixed_alignment_metadata = {
            "source_score": str(args.alignment_from_score),
            "source_reference_label": development["reference_label"],
            "source_scoring_start_realtime_ns": development[
                "scoring_start_realtime_ns"
            ],
        }
    report = score_treatments(
        args.treatments,
        args.motive,
        scoring_start_realtime_ns=boundary,
        front_plane_to_pelvis_x_m=args.front_plane_to_pelvis_x_m,
        reference_label=args.reference_label,
        estimator_to_reference_rotation=rotation,
        fixed_alignment_metadata=fixed_alignment_metadata,
        score_window_evaluator_alignment=args.score_window_evaluator_alignment,
        fixed_event_time_shift_ms=args.fixed_event_time_shift_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(args.output), "treatments": sorted(report["treatments"])},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
