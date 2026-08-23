#!/usr/bin/env python3
"""Render the decision figures for the G1-4123 localization investigation.

All plotted values are read from frozen JSON artifacts.  This script does not
recompute scores and never reads Motive as an online localization input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BLUE = "#1769AA"
CYAN = "#00A6A6"
GREEN = "#2E8B57"
ORANGE = "#E07A1F"
RED = "#C94242"
INK = "#1C2630"
MUTED = "#5C6773"
GRID = "#DCE3E8"
PAPER = "#F7F9FB"


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def primary(score: dict, treatment: str) -> dict[str, float]:
    row = score["aggregates"][treatment]["primary"]
    return {
        "position": float(row["planar_rmse_m"]["mean"]),
        "rpe": float(row["rpe_1s_translation_rmse_m"]["mean"]),
        "yaw": float(row["yaw_rmse_deg"]["mean"]),
        "availability": float(row["availability_fraction"]["mean"]),
    }


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=200, bbox_inches="tight", facecolor=PAPER)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight", facecolor=PAPER)


def decision_figure(artifact: Path, output_dir: Path) -> None:
    baseline_files = {
        "Walk 02": artifact / "data/scores/baseline-superodom-walk02-mesh-front.json",
        "Walk 03": artifact / "data/scores/baseline-superodom-walk03-mesh-front.json",
    }
    winner_files = {
        "Walk 02": artifact
        / "data/scores/kiss-icp-walk02-v015-exact-gyro-deskew-fk-livox-yaw-mesh-front.json",
        "Walk 03": artifact
        / "data/scores/kiss-icp-walk03-v015-exact-gyro-deskew-fk-livox-yaw-mesh-front.json",
    }
    baseline = {
        run: primary(load(path), "superodom_dynamic_fk_pelvis")
        for run, path in baseline_files.items()
    }
    winner_treatment = "kiss_icp_exact_gyro_deskew_fk_position_livox_gyro_torso_yaw"
    winner = {run: primary(load(path), winner_treatment) for run, path in winner_files.items()}

    contribution_no_fk = load(
        artifact / "data/scores/kiss-icp-walk02-v015-exact-gyro-deskew-livox-yaw.json"
    )
    contribution_with_fk = load(
        artifact / "data/scores/kiss-icp-walk02-v015-exact-gyro-deskew-fk-livox-yaw.json"
    )
    contribution_with_lever = load(winner_files["Walk 02"])
    contribution_rows = [
        ("Gio SuperOdom\n+ dynamic FK", baseline["Walk 02"]),
        (
            "KISS + exact\ninertial deskew",
            primary(contribution_no_fk, "kiss_icp_sensor"),
        ),
        (
            "+ Livox navigation\nheading",
            primary(
                contribution_no_fk,
                "kiss_icp_exact_gyro_deskew_position_livox_gyro_torso_yaw",
            ),
        ),
        (
            "+ dynamic\nwaist FK",
            primary(
                contribution_with_fk,
                "kiss_icp_exact_gyro_deskew_fk_position_livox_gyro_torso_yaw",
            ),
        ),
        (
            "+ measured front-plane\nlever arm",
            primary(contribution_with_lever, winner_treatment),
        ),
    ]

    contract = load(
        artifact / "data/scores/robot-vlm-two-lane-contract-front-offset-mesh-front.json"
    )
    cross = [
        ("02 fit → 03 eval", contract["cross_run"]["walk02_to_walk03"]),
        ("03 fit → 02 eval", contract["cross_run"]["walk03_to_walk02"]),
    ]
    map_audit = load(artifact / "data/map/polycam-walk02-translation-signal-audit-mesh-front.json")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )
    fig = plt.figure(figsize=(16, 9), facecolor=PAPER)
    grid = fig.add_gridspec(2, 2, left=0.06, right=0.97, top=0.84, bottom=0.08, wspace=0.24, hspace=0.38)
    fig.text(
        0.06,
        0.945,
        "G1-4123 localization: local accuracy improved ~24×; map translation was rejected",
        fontsize=22,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.895,
        "Two frozen AMO walks · Motive used only for scoring · selected output is pelvis x/y plus navigation-forward yaw",
        fontsize=12,
        color=MUTED,
    )

    # A: component ladder.
    ax = fig.add_subplot(grid[0, 0])
    x = np.arange(len(contribution_rows))
    pos = np.array([row[1]["position"] for row in contribution_rows])
    yaw = np.array([row[1]["yaw"] for row in contribution_rows])
    ax.set_title("A  What each component changed (Walk 02)", loc="left", pad=12)
    ax.plot(x, pos, "o-", color=BLUE, lw=2.5, ms=7, label="Planar RMSE (m)")
    ax.plot(x, yaw, "o-", color=ORANGE, lw=2.5, ms=7, label="Yaw RMSE (deg)")
    ax.set_yscale("log")
    ax.set_ylim(0.025, 180)
    ax.set_xticks(x, [row[0] for row in contribution_rows], fontsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper right")
    for i, value in enumerate(pos):
        ax.annotate(f"{value:.3f} m", (i, value), xytext=(0, -17), textcoords="offset points", ha="center", color=BLUE, fontsize=8)
    for i, value in enumerate(yaw):
        ax.annotate(f"{value:.1f}°", (i, value), xytext=(0, 8), textcoords="offset points", ha="center", color=ORANGE, fontsize=8)

    # B: independent repeat.
    ax = fig.add_subplot(grid[0, 1])
    ax.set_title("B  Improvement replicated on the held-out stress walk", loc="left", pad=12)
    runs = list(baseline)
    idx = np.arange(len(runs))
    width = 0.34
    base_pos = [baseline[r]["position"] for r in runs]
    win_pos = [winner[r]["position"] for r in runs]
    ax.bar(idx - width / 2, base_pos, width, color="#9AA6B2", label="Gio SuperOdom")
    ax.bar(idx + width / 2, win_pos, width, color=GREEN, label="Selected local lane")
    ax.set_yscale("log")
    ax.set_ylim(0.03, 2.5)
    ax.set_ylabel("Planar RMSE (m, log scale)")
    ax.set_xticks(idx, runs)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper right")
    for i, (b, w) in enumerate(zip(base_pos, win_pos)):
        ax.text(i - width / 2, b * 1.08, f"{b:.2f} m", ha="center", fontsize=9, color=MUTED)
        ax.text(i + width / 2, w * 0.83, f"{w:.3f} m", ha="center", va="top", fontsize=9, color=GREEN, fontweight="bold")
        ax.text(i, 0.035, f"{b / w:.0f}× lower", ha="center", fontsize=9, color=INK, fontweight="bold")
    yaw_note = "Yaw RMSE: " + " · ".join(
        f"{r[-2:]} {baseline[r]['yaw']:.1f}° → {winner[r]['yaw']:.1f}°" for r in runs
    )
    ax.text(0.0, 1.02, yaw_note, transform=ax.transAxes, fontsize=9, color=ORANGE)

    # C: contract scorecard.
    ax = fig.add_subplot(grid[1, 0])
    ax.set_title("C  robot-vlm contract: cross-run map placement", loc="left", pad=12)
    ax.axis("off")
    columns = ["Metric", "Target"] + [name for name, _ in cross]
    rows = [
        ("Position RMSE", "≤ 0.10 m", [f"{d['position_rmse_m']:.3f} m" for _, d in cross], [d["position_rmse_m"] <= 0.10 for _, d in cross]),
        ("Position p95", "≤ 0.20 m", [f"{d['position_p95_m']:.3f} m" for _, d in cross], [d["position_p95_m"] <= 0.20 for _, d in cross]),
        ("Yaw RMSE", "≤ 3°", [f"{d['yaw_rmse_deg']:.2f}°" for _, d in cross], [d["yaw_rmse_deg"] <= 3.0 for _, d in cross]),
        ("Yaw p95", "≤ 7°", [f"{d['yaw_p95_deg']:.2f}°" for _, d in cross], [d["yaw_p95_deg"] <= 7.0 for _, d in cross]),
        ("Availability", "≥ 99%", [f"{100*d['availability_at_quality_ge_0_5']:.1f}%" for _, d in cross], [d["availability_at_quality_ge_0_5"] >= 0.99 for _, d in cross]),
        ("False healthy", "< 1%", [f"{100*d['false_healthy_fraction_all_samples']:.2f}%" for _, d in cross], [d["false_healthy_fraction_all_samples"] < 0.01 for _, d in cross]),
        ("Pose age p95", "≤ 0.10 s", [f"{1000*d['pose_age_s']['p95']:.1f} ms" for _, d in cross], [d["pose_age_s"]["p95"] <= 0.10 for _, d in cross]),
    ]
    table = ax.table(
        cellText=[[metric, target, *values] for metric, target, values, _ in rows],
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.02, 1.0, 0.91],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(PAPER)
        if row == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("white" if row % 2 else "#EEF3F6")
            if col == 0:
                cell.get_text().set_ha("left")
                cell.get_text().set_weight("bold")
            if col >= 2:
                passed = rows[row - 1][3][col - 2]
                cell.get_text().set_color(GREEN if passed else RED)
                cell.get_text().set_weight("bold")
    ax.text(
        0,
        -0.03,
        "Two narrow misses remain: 0.117 m cross-run RMSE and 3.36° yaw RMSE in one direction.",
        transform=ax.transAxes,
        fontsize=9,
        color=RED,
        fontweight="bold",
    )

    # D: map translation negative result.
    ax = fig.add_subplot(grid[1, 1])
    ax.set_title("D  Why map translation is initialize-once", loc="left", pad=12)
    labels = ["Hold initial x/y", "Apply each ICP x/y"]
    values = [map_audit["fixed_position_rmse_at_updates_m"], map_audit["candidate_position_rmse_at_updates_m"]]
    bars = ax.bar(labels, values, color=[GREEN, RED], width=0.56)
    ax.axhline(0.10, color=MUTED, linestyle="--", linewidth=1.2, label="0.10 m RMSE target")
    ax.set_ylim(0, 0.24)
    ax.set_ylabel("Position RMSE at map updates (m)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper left")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f} m", ha="center", fontweight="bold")
    ax.text(
        0.02,
        0.73,
        f"Raw ICP translation helped only\n{100*map_audit['raw_candidate_improves_fraction']:.1f}% of updates.",
        transform=ax.transAxes,
        fontsize=11,
        color=RED,
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.53,
        "Decision: latch initial x/y;\naccept quality-gated heading updates.",
        transform=ax.transAxes,
        fontsize=11,
        color=INK,
    )

    save(fig, output_dir, "g1_4123_localization_decision_summary")
    plt.close(fig)


def architecture_figure(output_dir: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK})
    fig, ax = plt.subplots(figsize=(16, 7.6), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.text(
        0.06,
        0.93,
        "Selected architecture: fast local motion, conservative map anchoring, fail-closed output",
        fontsize=21,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.875,
        "Motive is outside the runtime graph; it appears only below the evidence boundary as evaluator ground truth.",
        fontsize=12,
        color=MUTED,
    )

    def box(x: float, y: float, w: float, h: float, title: str, body: str, color: str, *, dashed: bool = False) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            linewidth=1.8,
            edgecolor=color,
            facecolor="white",
            linestyle="--" if dashed else "-",
        )
        ax.add_patch(patch)
        ax.text(x + 0.18, y + h - 0.33, title, fontsize=11, fontweight="bold", color=color, va="top")
        ax.text(x + 0.18, y + h - 0.75, body, fontsize=9.3, color=INK, va="top", linespacing=1.35)

    def arrow(x1: float, y1: float, x2: float, y2: float, color: str = MUTED, *, dashed: bool = False) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.6,
                color=color,
                linestyle="--" if dashed else "-",
            )
        )

    ax.text(0.5, 6.75, "LOCAL LANE · every Livox scan", fontsize=10, fontweight="bold", color=BLUE)
    box(0.5, 4.65, 2.2, 1.65, "Onboard inputs", "Livox points + point time\nLivox IMU\nG1 qpos + timestamps", BLUE)
    box(3.25, 4.65, 2.35, 1.65, "Exact deskew", "Bias-corrected 3-axis\nrotational integration\nco-mounted IMU", BLUE)
    box(6.15, 4.65, 2.1, 1.65, "KISS-ICP", "0.15 m voxel\n0.5–15 m range\n~7–8 ms mean", BLUE)
    box(8.8, 4.65, 2.35, 1.65, "Frame contract", "Dynamic waist FK → pelvis x/y\nLivox gyro → navigation yaw\npose, twist, covariance", BLUE)
    arrow(2.7, 5.48, 3.25, 5.48, BLUE)
    arrow(5.6, 5.48, 6.15, 5.48, BLUE)
    arrow(8.25, 5.48, 8.8, 5.48, BLUE)

    ax.text(0.5, 3.78, "MAP LANE · slower and quality gated", fontsize=10, fontweight="bold", color=CYAN)
    box(0.5, 1.75, 2.2, 1.55, "Polycam scan", "Vertical-persistence\nstructural point map\nID + digest pinned", CYAN)
    box(3.25, 1.75, 2.35, 1.55, "Map registration", "Initial all-geometry fit\nfitness / RMSE / p95\nobservability checks", CYAN)
    box(6.15, 1.75, 2.1, 1.55, "Policy", "Latch initial x/y\nHeading-only updates\n15 s hard hold", CYAN)
    arrow(2.7, 2.52, 3.25, 2.52, CYAN)
    arrow(5.6, 2.52, 6.15, 2.52, CYAN)

    box(12.0, 3.25, 3.1, 2.35, "robot-vlm adapter", "base_pose = (x, y, yaw)\nloc_quality ∈ [0, 1]\n0 on stale/reset/failed gate\ndrive floor = 0.5\npose hard stale = 0.3 s", GREEN)
    arrow(11.15, 5.48, 12.0, 4.75, GREEN)
    arrow(8.25, 2.52, 12.0, 4.05, GREEN)

    ax.plot([0.4, 15.4], [1.15, 1.15], color=RED, linestyle="--", linewidth=1.5)
    ax.text(0.5, 0.87, "EVALUATION-ONLY BOUNDARY", fontsize=9.5, fontweight="bold", color=RED)
    box(4.5, 0.15, 3.1, 0.7, "Motive front-pelvis rigid body", "cross-run score + false-healthy audit; never runtime input", RED, dashed=True)
    arrow(6.05, 0.85, 6.05, 1.13, RED, dashed=True)
    ax.text(
        10.0,
        0.55,
        "Next gate: passive live shadow publisher\nwith proven LiDAR/IMU/common-clock semantics",
        fontsize=10,
        color=RED,
        fontweight="bold",
        va="center",
    )

    save(fig, output_dir, "g1_4123_localization_selected_architecture")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    decision_figure(args.artifact_dir, args.output_dir)
    architecture_figure(args.output_dir)


if __name__ == "__main__":
    main()
