#!/usr/bin/env python3
"""Render and tabulate frozen-window AMO localization scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SENSOR = "superodom_sensor"
DYNAMIC_FK = "superodom_dynamic_fk_pelvis"
TREATMENT_LABELS = {
    SENSOR: "LiDAR sensor",
    DYNAMIC_FK: "dynamic-FK pelvis",
    "superodom_dynamic_fk_position_lio_heading_root_gravity": "FK + root gravity",
}
CONTENT_COLORS = {
    "dynamic_translation_turning": "#2878B5",
    "yaw_stress": "#D95319",
    "stationary": "#2CA02C",
}


def parse_score(value: str) -> tuple[str, Path]:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("score must be LABEL=PATH") from error
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("score must be LABEL=PATH")
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"score does not exist: {path}")
    return label, path


def load_scores(specifications: list[tuple[str, Path]]) -> list[dict[str, object]]:
    loaded: list[dict[str, object]] = []
    for label, path in specifications:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "g1_amo_segmented_localization_score_v1":
            raise ValueError(f"unsupported score schema in {path}")
        payload["display_label"] = label
        payload["source_path"] = str(path)
        loaded.append(payload)
    return loaded


def write_metrics(scores: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "run",
            "treatment",
            "admission",
            "content",
            "segment_count",
            "availability_mean",
            "planar_rmse_mean_m",
            "planar_rmse_p50_m",
            "planar_rmse_p95_m",
            "rpe_1s_translation_rmse_mean_m",
            "yaw_rmse_mean_deg",
            "yaw_rmse_p95_deg",
            "alignment_translation_rmse_m",
            "alignment_translation_p95_m",
            "alignment_rotation_deg",
            "deployable_global_localization_claim_admitted",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for score in scores:
            alignment = score["alignment"]
            for treatment, treatment_metrics in score["aggregates"].items():
                groups = [("primary", "all", treatment_metrics["primary"])]
                groups.extend(
                    ("all", content, metrics)
                    for content, metrics in treatment_metrics["by_content"].items()
                )
                groups.append(
                    ("masked_secondary", "all", treatment_metrics["masked_secondary"])
                )
                for admission, content, metrics in groups:
                    writer.writerow(
                        {
                            "run": score["display_label"],
                            "treatment": treatment,
                            "admission": admission,
                            "content": content,
                            "segment_count": metrics["segment_count"],
                            "availability_mean": metrics["availability_fraction"]["mean"],
                            "planar_rmse_mean_m": metrics["planar_rmse_m"]["mean"],
                            "planar_rmse_p50_m": metrics["planar_rmse_m"]["p50"],
                            "planar_rmse_p95_m": metrics["planar_rmse_m"]["p95"],
                            "rpe_1s_translation_rmse_mean_m": metrics[
                                "rpe_1s_translation_rmse_m"
                            ]["mean"],
                            "yaw_rmse_mean_deg": metrics["yaw_rmse_deg"]["mean"],
                            "yaw_rmse_p95_deg": metrics["yaw_rmse_deg"]["p95"],
                            "alignment_translation_rmse_m": alignment[
                                "translation_rmse_m"
                            ],
                            "alignment_translation_p95_m": alignment[
                                "translation_p95_m"
                            ],
                            "alignment_rotation_deg": alignment["rotation_deg"],
                            "deployable_global_localization_claim_admitted": alignment[
                                "deployable_global_localization_claim_admitted"
                            ],
                        }
                    )


def _sensor_segments(score: dict[str, object]) -> list[dict[str, object]]:
    return [row for row in score["segments"] if row["treatment"] == SENSOR]


def _dynamic_fk_by_window(score: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        row["window_id"]: row
        for row in score["segments"]
        if row["treatment"] == DYNAMIC_FK
    }


def render(scores: list[dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.5), constrained_layout=True)
    panels = (
        (axes[0, 0], "planar_error_m", "rmse", "Window-relative planar RMSE (m)"),
        (axes[0, 1], "yaw_error_deg", "rmse", "Window-relative yaw RMSE (deg)"),
        (
            axes[1, 0],
            "rpe",
            "translation_error_m",
            "1 s relative translation RMSE (m)",
        ),
    )
    legend_handles: dict[str, object] = {}
    for run_index, score in enumerate(scores):
        rows = _sensor_segments(score)
        x = np.arange(len(rows), dtype=float) + run_index * (len(rows) + 1.5)
        labels = [f"{score['display_label']}\n{row['window_id']}" for row in rows]
        for axis, metric, leaf, title in panels:
            if metric == "rpe":
                values = [row["metrics"]["rpe"]["1.0"][leaf]["rmse"] for row in rows]
            else:
                values = [row["metrics"][metric][leaf] for row in rows]
            colors = [CONTENT_COLORS[row["content"]] for row in rows]
            bars = axis.bar(x, values, width=0.78, color=colors, edgecolor="#222222")
            for bar, row in zip(bars, rows):
                if row["admission"] != "primary":
                    bar.set_hatch("///")
                    bar.set_alpha(0.55)
                legend_handles.setdefault(row["content"], bar)
            axis.set_xticks(x, labels, rotation=55, ha="right", fontsize=7)
            axis.set_title(title, loc="left", fontweight="bold")
            axis.grid(axis="y", alpha=0.25)

        dynamic_fk = _dynamic_fk_by_window(score)
        deltas = [
            dynamic_fk[row["window_id"]]["metrics"]["planar_error_m"]["rmse"]
            - row["metrics"]["planar_error_m"]["rmse"]
            for row in rows
        ]
        colors = ["#2CA02C" if delta < 0 else "#D95319" for delta in deltas]
        bars = axes[1, 1].bar(x, deltas, width=0.78, color=colors, edgecolor="#222222")
        for bar, row in zip(bars, rows):
            if row["admission"] != "primary":
                bar.set_hatch("///")
                bar.set_alpha(0.55)
        axes[1, 1].set_xticks(x, labels, rotation=55, ha="right", fontsize=7)

    axes[1, 1].axhline(0.0, color="black", linewidth=1)
    axes[1, 1].set_title(
        "Dynamic-FK minus sensor planar RMSE (m; negative is better)",
        loc="left",
        fontweight="bold",
    )
    axes[1, 1].grid(axis="y", alpha=0.25)
    fig.suptitle(
        "G1-4123 stress datasets isolate locomotion/yaw failure\n"
        "Fixed 5–15 s Motive windows; hatched bars are masked-secondary sensitivity windows",
        fontsize=16,
        fontweight="bold",
    )
    fig.legend(
        list(legend_handles.values()),
        [key.replace("_", " ") for key in legend_handles],
        loc="outside lower center",
        ncol=len(legend_handles),
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", action="append", type=parse_score, required=True)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    args = parser.parse_args()
    scores = load_scores(args.score)
    write_metrics(scores, args.output_metrics)
    render(scores, args.output_figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
