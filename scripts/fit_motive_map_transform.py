#!/usr/bin/env python3
"""Fit a metric, right-handed Motive(Y-up) to robot-vlm map(Z-up) transform."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


SCHEMA = "g1_motive_polycam_transform_v1"


def _content_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _reflective_planar_fit(motive_xz: np.ndarray, map_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return map_xy = A @ motive_xz + t with det(A) fixed to -1.

    Motive is right-handed Y-up and robot-vlm is right-handed Z-up.  Preserving
    both positive up axes therefore requires a reflection in the horizontal
    XZ->XY projection; allowing the +1 branch recreates the historical mirrored
    path ambiguity.
    """
    motive_xz = np.asarray(motive_xz, dtype=np.float64)
    map_xy = np.asarray(map_xy, dtype=np.float64)
    if motive_xz.shape != map_xy.shape or motive_xz.ndim != 2 or motive_xz.shape[1] != 2:
        raise ValueError("planar correspondence arrays must both be Nx2")
    if len(motive_xz) < 2:
        raise ValueError("at least two planar correspondences are required")
    p_mean, q_mean = motive_xz.mean(axis=0), map_xy.mean(axis=0)
    H = (motive_xz - p_mean).T @ (map_xy - q_mean)
    U, _, Vt = np.linalg.svd(H)
    A = Vt.T @ U.T
    if np.linalg.det(A) > 0.0:
        Vt[-1] *= -1.0
        A = Vt.T @ U.T
    if not np.isclose(np.linalg.det(A), -1.0, atol=1e-8):
        raise ValueError("failed to construct the right-handed Y-up to Z-up transform")
    return A, q_mean - A @ p_mean


def _full_transform(A: np.ndarray, t_xy: np.ndarray, t_z: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        (
            (A[0, 0], 0.0, A[0, 1]),
            (A[1, 0], 0.0, A[1, 1]),
            (0.0, 1.0, 0.0),
        )
    )
    transform[:2, 3] = t_xy
    transform[2, 3] = float(t_z)
    if not np.isclose(np.linalg.det(transform[:3, :3]), 1.0, atol=1e-8):
        raise ValueError("Motive-to-map rotation is not proper")
    return transform


def _apply(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points_h = np.column_stack((points, np.ones(len(points))))
    return (transform @ points_h.T).T[:, :3]


def _pairwise_scale(motive_xyz: np.ndarray, map_xyz: np.ndarray) -> np.ndarray:
    ratios = []
    for i, j in itertools.combinations(range(len(motive_xyz)), 2):
        source = float(np.linalg.norm(motive_xyz[i] - motive_xyz[j]))
        target = float(np.linalg.norm(map_xyz[i] - map_xyz[j]))
        if source > 0.15:
            ratios.append(target / source)
    return np.asarray(ratios, dtype=np.float64)


def fit_control_points(value: dict, *, inlier_threshold_m: float = 0.07) -> dict:
    if value.get("schema") != "g1_motive_polycam_control_points_v1":
        raise ValueError("unsupported control-point schema")
    identity = value.get("map_identity", {})
    required_identity = ("glb_sha256", "surface_sha256", "structural_map_sha256")
    if any(len(str(identity.get(key, ""))) != 64 for key in required_identity):
        raise ValueError("control points are not bound to all map digests")
    contract = value.get("motive_contract", {})
    if contract.get("up_axis") != "y" or contract.get("horizontal_axes") != ["x", "z"]:
        raise ValueError("Motive contract must be right-handed Y-up with horizontal x/z")

    points = value.get("points", [])
    fit_rows = [row for row in points if row.get("role") == "fit"]
    holdout_rows = [row for row in points if row.get("role") == "holdout"]
    if len(fit_rows) < 4 or len(holdout_rows) < 1:
        raise ValueError("need at least four fit points and one held-out point")
    labels = [str(row.get("label", "")) for row in points]
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError("control-point labels must be nonempty and unique")

    motive_fit = np.asarray([row["motive_xyz_m"] for row in fit_rows], dtype=np.float64)
    map_fit = np.asarray([row["map_xyz_m"] for row in fit_rows], dtype=np.float64)
    if motive_fit.shape != (len(fit_rows), 3) or map_fit.shape != motive_fit.shape:
        raise ValueError("every fit control point must have finite 3D Motive and map positions")
    if not np.all(np.isfinite(motive_fit)) or not np.all(np.isfinite(map_fit)):
        raise ValueError("control points must be finite")

    # Enumerate minimal subsets so one misidentified landmark cannot pull the
    # transform across the room.  Refit all inliers from the best hypothesis.
    best = None
    for subset in itertools.combinations(range(len(fit_rows)), 3):
        idx = np.asarray(subset, dtype=np.int64)
        A, t_xy = _reflective_planar_fit(motive_fit[idx][:, (0, 2)], map_fit[idx, :2])
        t_z = float(np.median(map_fit[idx, 2] - motive_fit[idx, 1]))
        transform = _full_transform(A, t_xy, t_z)
        residual = np.linalg.norm(_apply(transform, motive_fit) - map_fit, axis=1)
        inliers = residual <= inlier_threshold_m
        signature = (int(inliers.sum()), -float(np.sum(np.minimum(residual, inlier_threshold_m))))
        if best is None or signature > best[0]:
            best = (signature, inliers)
    assert best is not None
    inliers = best[1]
    if int(inliers.sum()) < 4:
        raise ValueError("fewer than four fit landmarks agree within the inlier threshold")
    A, t_xy = _reflective_planar_fit(motive_fit[inliers][:, (0, 2)], map_fit[inliers, :2])
    t_z = float(np.median(map_fit[inliers, 2] - motive_fit[inliers, 1]))
    transform = _full_transform(A, t_xy, t_z)

    motive_all = np.asarray([row["motive_xyz_m"] for row in points], dtype=np.float64)
    map_all = np.asarray([row["map_xyz_m"] for row in points], dtype=np.float64)
    residual_all = np.linalg.norm(_apply(transform, motive_all) - map_all, axis=1)
    fit_residual = residual_all[: len(fit_rows)] if points[: len(fit_rows)] == fit_rows else np.asarray(
        [residual_all[points.index(row)] for row in fit_rows]
    )
    holdout_residual = np.asarray([residual_all[points.index(row)] for row in holdout_rows])
    scale = _pairwise_scale(motive_fit[inliers], map_fit[inliers])
    scale_median = float(np.median(scale)) if len(scale) else float("nan")

    # Leave-one-out exposes a landmark layout that is numerically fragile even
    # when the all-point residual looks small.
    loo_errors = []
    for omitted in range(len(fit_rows)):
        keep = np.arange(len(fit_rows)) != omitted
        A_loo, t_loo = _reflective_planar_fit(motive_fit[keep][:, (0, 2)], map_fit[keep, :2])
        tz_loo = float(np.median(map_fit[keep, 2] - motive_fit[keep, 1]))
        prediction = _apply(_full_transform(A_loo, t_loo, tz_loo), motive_fit[[omitted]])[0]
        loo_errors.append(float(np.linalg.norm(prediction - map_fit[omitted])))

    metrics = {
        "fit_rmse_m": float(np.sqrt(np.mean(fit_residual**2))),
        "fit_p95_m": float(np.quantile(fit_residual, 0.95)),
        "fit_max_m": float(np.max(fit_residual)),
        "holdout_rmse_m": float(np.sqrt(np.mean(holdout_residual**2))),
        "holdout_max_m": float(np.max(holdout_residual)),
        "leave_one_out_max_m": float(np.max(loo_errors)),
        "pairwise_scale_median": scale_median,
        "pairwise_scale_p95_abs_error": float(np.quantile(np.abs(scale - 1.0), 0.95)),
        "inlier_count": int(inliers.sum()),
        "fit_count": len(fit_rows),
        "holdout_count": len(holdout_rows),
        "horizontal_determinant": float(np.linalg.det(A)),
        "rotation_determinant": float(np.linalg.det(transform[:3, :3])),
        "up_alignment_dot": float(transform[:3, :3][:, 1] @ np.asarray((0.0, 0.0, 1.0))),
    }
    gates = {
        "maximum_fit_rmse_m": 0.04,
        "maximum_fit_p95_m": 0.06,
        "maximum_holdout_m": 0.08,
        "maximum_leave_one_out_m": 0.10,
        "minimum_pairwise_scale": 0.98,
        "maximum_pairwise_scale": 1.02,
    }
    failures = []
    if metrics["fit_rmse_m"] > gates["maximum_fit_rmse_m"]:
        failures.append("fit_rmse")
    if metrics["fit_p95_m"] > gates["maximum_fit_p95_m"]:
        failures.append("fit_p95")
    if metrics["holdout_max_m"] > gates["maximum_holdout_m"]:
        failures.append("holdout")
    if metrics["leave_one_out_max_m"] > gates["maximum_leave_one_out_m"]:
        failures.append("leave_one_out")
    if not gates["minimum_pairwise_scale"] <= scale_median <= gates["maximum_pairwise_scale"]:
        failures.append("metric_scale")

    result = {
        "schema": SCHEMA,
        "status": "accepted" if not failures else "rejected",
        "command_capability": "structurally_unavailable",
        "map_identity": identity,
        "motive_contract": contract,
        "T_map_from_motive": transform.tolist(),
        "fit_inlier_labels": [row["label"] for row, keep in zip(fit_rows, inliers) if keep],
        "residuals_m": {row["label"]: float(error) for row, error in zip(points, residual_all)},
        "metrics": metrics,
        "gates": gates,
        "failures": failures,
        "control_points_sha256": _content_sha256(value),
    }
    result["content_sha256"] = _content_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-points", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inlier-threshold-m", type=float, default=0.07)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite: {args.output}")
    value = json.loads(args.control_points.read_text(encoding="utf-8"))
    result = fit_control_points(value, inlier_threshold_m=args.inlier_threshold_m)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "accepted" else 4)


if __name__ == "__main__":
    main()
