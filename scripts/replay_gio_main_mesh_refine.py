#!/usr/bin/env python3
"""Replay Gio-main's continuous Polycam mesh refine from a common map anchor."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _pose(position: object, quaternion: object) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(
        np.asarray(quaternion, dtype=np.float64)
    ).as_matrix()
    result[:3, 3] = np.asarray(position, dtype=np.float64)
    return result


def yaw(transform: np.ndarray) -> float:
    return float(np.arctan2(transform[1, 0], transform[0, 0]))


def _load_current(
    path: Path, topic: str, treatment: str | None
) -> tuple[np.ndarray, np.ndarray]:
    time_ns, poses = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        is_odometry = row.get("kind") == "odometry" and row.get("topic") == topic
        is_treatment = row.get("kind") == "pose" and row.get("treatment") == treatment
        if not (is_odometry or is_treatment):
            continue
        time_ns.append(int(row.get("source_time_ns", row.get("event_realtime_ns"))))
        poses.append(_pose(row["position_xyz_m"], row["quaternion_xyzw"]))
    if len(poses) < 3:
        raise ValueError(f"missing current odometry topic {topic!r}")
    return np.asarray(time_ns, dtype=np.int64), np.asarray(poses)


def _load_gio(
    path: Path, treatment: str | None, odometry_topic: str | None
) -> tuple[np.ndarray, np.ndarray]:
    time_ns, poses = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        is_treatment = row.get("kind") == "pose" and row.get("treatment") == treatment
        is_odometry = row.get("kind") == "odometry" and row.get("topic") == odometry_topic
        if not (is_treatment or is_odometry):
            continue
        time_ns.append(int(row.get("event_realtime_ns", row.get("source_time_ns"))))
        poses.append(_pose(row["position_xyz_m"], row["quaternion_xyzw"]))
    if len(poses) < 3:
        raise ValueError(f"missing Gio treatment {treatment!r}")
    return np.asarray(time_ns, dtype=np.int64), np.asarray(poses)


def _nearest(time_ns: np.ndarray, poses: np.ndarray, query_ns: int) -> np.ndarray:
    index = int(np.searchsorted(time_ns, query_ns, side="left"))
    index = min(max(index, 0), len(time_ns) - 1)
    if index > 0 and abs(int(time_ns[index - 1]) - query_ns) < abs(
        int(time_ns[index]) - query_ns
    ):
        index -= 1
    return poses[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gio-main-root", type=Path, required=True)
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--initialization-receipt", type=Path, required=True)
    parser.add_argument("--current-odometry", type=Path, required=True)
    parser.add_argument("--current-topic", default="/g1/localization/pelvis_odom")
    parser.add_argument("--current-treatment")
    parser.add_argument("--gio-treatment-track", type=Path, required=True)
    parser.add_argument("--gio-treatment", default="superodom_dynamic_fk_pelvis")
    parser.add_argument("--gio-odometry-topic")
    parser.add_argument("--registered-scans", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refine-hz", type=float, default=2.0)
    parser.add_argument("--sample-count", type=int, default=200_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.refine_hz <= 0.0:
        parser.error("refine frequency must be positive")

    module_dir = args.gio_main_root / "deploy/g1/localize"
    sys.path.insert(0, str(module_dir))
    from localizer import Localizer  # type: ignore
    from mesh_map import load_mesh_map  # type: ignore

    receipt = json.loads(args.initialization_receipt.read_text(encoding="utf-8"))
    init_ns = int(receipt["scan"]["evidence_time_ns"])
    map_T_current = np.asarray(receipt["result"]["T_map_local"], dtype=np.float64)
    current_time, current_pose = _load_current(
        args.current_odometry, args.current_topic, args.current_treatment
    )
    gio_time, gio_pose = _load_gio(
        args.gio_treatment_track,
        args.gio_treatment if args.gio_odometry_topic is None else None,
        args.gio_odometry_topic,
    )
    common_map_T_pelvis = map_T_current @ _nearest(
        current_time, current_pose, init_ns
    )
    gio_at_anchor = _nearest(gio_time, gio_pose, init_ns)
    yaw_delta = yaw(common_map_T_pelvis) - yaw(gio_at_anchor)
    rotation = Rotation.from_rotvec((0.0, 0.0, yaw_delta)).as_matrix()
    map_T_gio = np.eye(4, dtype=np.float64)
    map_T_gio[:3, :3] = rotation
    map_T_gio[:2, 3] = (
        common_map_T_pelvis[:2, 3] - rotation[:2, :2] @ gio_at_anchor[:2, 3]
    )
    map_T_gio[2, 3] = common_map_T_pelvis[2, 3] - gio_at_anchor[2, 3]

    # Gio-main sampled the mesh at process startup. Fixing NumPy's seed makes
    # the offline reproduction deterministic without changing its algorithm.
    np.random.seed(0)
    mesh = load_mesh_map(
        str(args.glb), up="y", sample_count=args.sample_count, verbose=False
    )
    localizer = Localizer(mesh, dof="4dof")
    localizer.d_max_refine = 0.15
    localizer.set_manual_guess(map_T_gio)

    with np.load(args.registered_scans, allow_pickle=False) as archive:
        points = np.asarray(archive["points_xyz_m"], dtype=np.float32)
        offsets = np.asarray(archive["cloud_offsets"], dtype=np.int64)
        scan_time = np.asarray(archive["source_time_ns"], dtype=np.int64)
    period_ns = round(1e9 / args.refine_hz)
    last_refine_ns = init_ns - period_ns
    rows = []
    for index, time_ns in enumerate(scan_time):
        time_ns = int(time_ns)
        if time_ns < init_ns or time_ns - last_refine_ns < period_ns:
            continue
        localizer.update_odom(_nearest(gio_time, gio_pose, time_ns))
        health = localizer.refine(points[offsets[index] : offsets[index + 1]])
        last_refine_ns = time_ns
        rows.append(
            {
                "schema": "g1_gio_main_mesh_refine_replay_v1",
                "kind": "map_transform",
                "source_time_ns": time_ns,
                "T_map_local": localizer.T_map_ci().tolist(),
                "status": health.status,
                "fitness": float(health.fitness),
                "rmse_m": float(health.rmse),
                "min_eig": float(health.min_eig),
                "cond_number": float(health.cond_number),
                "n_iter": int(health.n_iter),
            }
        )
    if not rows:
        raise ValueError("no Gio-main refinement step was formed")
    metadata = {
        "schema": "g1_gio_main_mesh_refine_replay_v1",
        "kind": "metadata",
        "gio_main_root": str(args.gio_main_root),
        "map_anchor_time_ns": init_ns,
        "initial_T_map_local": map_T_gio.tolist(),
        "sample_count": args.sample_count,
        "sample_seed": 0,
        "refine_hz": args.refine_hz,
        "step_count": len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metadata, sort_keys=True)
        + "\n"
        + "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "step_count": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
