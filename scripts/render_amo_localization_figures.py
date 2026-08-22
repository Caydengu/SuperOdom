#!/usr/bin/env python3
"""Render the frozen AMO localization treatment and map diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def treatment_metric(report: dict[str, object], treatment: str, group: str) -> float:
    return float(report["treatments"][treatment][group]["rmse"])


def transform_query(report: dict[str, object], query: np.ndarray) -> np.ndarray:
    rotation = np.asarray(report["rotation_matrix"], dtype=np.float64)
    translation = np.asarray(report["translation_m"], dtype=np.float64)
    return query @ rotation + translation


def plot_treatments(run: Path, output: Path) -> None:
    development = load_json(
        run / "data/scores/walk-01-posthoc-hybrid-mesh-inferred-pelvis-relative.json"
    )
    holdout = load_json(
        run / "data/scores/walk-02-posthoc-hybrid-mesh-inferred-pelvis-relative.json"
    )
    mapped = load_json(
        run
        / "data/scores/walk-02-hybrid-fresh-map-gated-fixed-dev-prewalk-alignment.json"
    )
    treatments = [
        ("LIO sensor", "superodom_sensor"),
        ("Dynamic FK", "superodom_dynamic_fk_pelvis"),
        ("Gyro fusion", "superodom_dynamic_fk_pelvis_imu_sign_pos"),
        (
            "FK + LIO yaw\n+ root gravity",
            "superodom_dynamic_fk_position_lio_heading_root_gravity",
        ),
    ]
    mapped_name = "superodom_hybrid_root_fresh_map_gated"
    colors = ["#8b95a5", "#4c78a8", "#e45756", "#2ca58d"]

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), constrained_layout=True)
    x = np.arange(len(treatments))
    width = 0.36
    for group, axis, title, ylabel in (
        ("planar_error_m", axes[0], "Local pelvis trajectory", "Planar RMSE (m)"),
        ("yaw_error_deg", axes[1], "Pelvis heading", "Yaw RMSE (deg)"),
    ):
        dev = [treatment_metric(development, key, group) for _, key in treatments]
        test = [treatment_metric(holdout, key, group) for _, key in treatments]
        axis.bar(
            x - width / 2, dev, width, color=colors, alpha=0.62, label="walk-01 dev"
        )
        axis.bar(
            x + width / 2,
            test,
            width,
            color=colors,
            edgecolor="#16212b",
            label="walk-02 holdout",
        )
        axis.set_xticks(x, [label for label, _ in treatments], rotation=18, ha="right")
        axis.set_title(title, loc="left", weight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend(frameon=False, fontsize=9)

    map_values = [
        treatment_metric(holdout, "superodom_dynamic_fk_pelvis", "planar_error_m"),
        treatment_metric(mapped, mapped_name, "planar_error_m"),
    ]
    map_yaw = [
        treatment_metric(holdout, "superodom_dynamic_fk_pelvis", "yaw_error_deg"),
        treatment_metric(mapped, mapped_name, "yaw_error_deg"),
    ]
    mx = np.arange(2)
    axes[2].bar(
        mx - width / 2, map_values, width, color="#4c78a8", label="planar RMSE (m)"
    )
    yaw_axis = axes[2].twinx()
    yaw_axis.bar(
        mx + width / 2, map_yaw, width, color="#f2a541", label="yaw RMSE (deg)"
    )
    axes[2].set_xticks(mx, ["Local only", "Fresh-map gated"])
    axes[2].set_ylabel("Planar RMSE (m)", color="#4c78a8")
    yaw_axis.set_ylabel("Yaw RMSE (deg)", color="#c97713")
    axes[2].set_title("Deployable holdout result", loc="left", weight="bold")
    axes[2].set_xlabel(
        "0.88 s init · 99.2% availability · Motive evaluator-only", fontsize=9
    )
    axes[2].grid(axis="y", alpha=0.22)
    fig.suptitle(
        "G1 AMO localization: matched-input treatment results",
        fontsize=15,
        weight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_maps(run: Path, output: Path) -> None:
    fresh_report = load_json(
        run / "data/map-localization/fresh-map-walk-02-init-1s.json"
    )
    legacy_report = load_json(
        run / "data/map-localization/legacy-field-bay-walk-02-init-1s.json"
    )
    corrections = load_json(
        run / "data/map-localization/fresh-map-walk-02-gated-corrections.json"
    )
    with np.load(
        run / "data/map-localization/fresh-map-walk-02-init-1s-clouds.npz"
    ) as data:
        fresh_map = np.asarray(data["map_xy"])
        fresh_query = np.asarray(data["query_xy"])
    with np.load(
        run / "data/map-localization/legacy-field-bay-walk-02-init-1s-clouds.npz"
    ) as data:
        legacy_map = np.asarray(data["map_xy"])
        legacy_query = np.asarray(data["query_xy"])

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.8), constrained_layout=True)
    for axis, map_xy, query_xy, report, title in (
        (axes[0], fresh_map, fresh_query, fresh_report, "Fresh walk-01 map: admitted"),
        (
            axes[1],
            legacy_map,
            legacy_query,
            legacy_report,
            "Legacy Field Bay map: rejected",
        ),
    ):
        stride = max(1, map_xy.shape[0] // 9000)
        qstride = max(1, query_xy.shape[0] // 4000)
        axis.scatter(
            map_xy[::stride, 0],
            map_xy[::stride, 1],
            s=2,
            c="#9aa3ad",
            alpha=0.35,
            rasterized=True,
            label="map",
        )
        transformed = transform_query(report, query_xy)
        axis.scatter(
            transformed[::qstride, 0],
            transformed[::qstride, 1],
            s=3,
            c="#0d8f7a",
            alpha=0.6,
            rasterized=True,
            label="1 s query",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title, loc="left", weight="bold")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.grid(alpha=0.15)
        axis.text(
            0.02,
            0.98,
            f"yaw {float(report['yaw_deg']):.1f}°\nICP p95 {float(report['icp']['p95_m']):.3f} m\nhealthy {str(report['healthy']).lower()}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "alpha": 0.88,
                "edgecolor": "#cdd3d8",
            },
        )
    axes[0].legend(frameon=False, loc="lower right")

    accepted = [row for row in corrections["corrections"] if row["accepted"]]
    start = int(accepted[0]["available_after_source_ns"])
    seconds = [
        (int(row["available_after_source_ns"]) - start) * 1e-9 for row in accepted
    ]
    yaw = [float(row["yaw_deg"]) for row in accepted]
    p95 = [float(row["quality"]["p95_m"]) for row in accepted]
    axes[2].plot(seconds, yaw, color="#4c78a8", lw=2.0, label="map-to-odom yaw")
    axes[2].set_xlabel("time after map initialization (s)")
    axes[2].set_ylabel("map-to-odom yaw (deg)", color="#4c78a8")
    quality_axis = axes[2].twinx()
    quality_axis.plot(
        seconds, p95, color="#e45756", lw=1.7, alpha=0.85, label="ICP p95"
    )
    quality_axis.axhline(
        float(corrections["gates"]["maximum_p95_m"]),
        color="#e45756",
        ls="--",
        alpha=0.55,
    )
    quality_axis.set_ylabel("ICP p95 residual (m)", color="#b43b3a")
    axes[2].set_title("Causal correction confidence", loc="left", weight="bold")
    axes[2].grid(alpha=0.2)
    axes[2].text(
        0.02,
        0.12,
        f"{corrections['accepted_count']}/{corrections['attempt_count']} periodic updates admitted\nno Motive input",
        transform=axes[2].transAxes,
        va="bottom",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "#cdd3d8",
        },
    )
    fig.suptitle(
        "Map localization diagnostics on frozen walk-02", fontsize=15, weight="bold"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    treatment_path = args.output_dir / "localization-treatment-comparison.png"
    map_path = args.output_dir / "map-localization-diagnostics.png"
    plot_treatments(args.run_dir, treatment_path)
    plot_maps(args.run_dir, map_path)
    print(json.dumps({"outputs": [str(treatment_path), str(map_path)]}, sort_keys=True))


if __name__ == "__main__":
    main()
