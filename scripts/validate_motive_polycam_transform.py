#!/usr/bin/env python3
"""Validate an independent Motive-to-Polycam evaluator transform contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "g1_motive_polycam_transform_v1"


def content_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _determinant3(matrix: list[list[float]]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def validate_transform(
    value: dict[str, Any],
    *,
    expected_glb_sha256: str,
    expected_surface_sha256: str,
    expected_structural_map_sha256: str,
    expected_pelvis_rigid_body_id: int,
    expected_pelvis_rigid_body_name: str,
) -> dict[str, Any]:
    """Return an identity summary or raise ValueError on any contract mismatch."""
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported Motive/Polycam transform schema")
    recorded_digest = value.get("content_sha256")
    unsigned = {key: item for key, item in value.items() if key != "content_sha256"}
    observed_digest = content_sha256(unsigned)
    if recorded_digest != observed_digest:
        raise ValueError("Motive/Polycam transform content digest mismatch")
    if value.get("status") != "accepted" or value.get("evaluator_ready") is not True:
        raise ValueError("Motive/Polycam transform is not evaluator-ready")
    if value.get("command_capability") != "structurally_unavailable":
        raise ValueError("Motive evaluator artifact must be command-incapable")
    if value.get("failures") != []:
        raise ValueError("accepted Motive transform still reports failed gates")

    expected_identity = {
        "glb_sha256": expected_glb_sha256.lower(),
        "surface_sha256": expected_surface_sha256.lower(),
        "structural_map_sha256": expected_structural_map_sha256.lower(),
    }
    if value.get("map_identity") != expected_identity:
        raise ValueError("Motive transform belongs to a different Polycam map identity")
    motive_contract = value.get("motive_contract", {})
    if motive_contract.get("up_axis") != "y" or motive_contract.get("horizontal_axes") != ["x", "z"]:
        raise ValueError("Motive transform does not declare the required Y-up x/z contract")

    transform = value.get("T_map_from_motive")
    if not isinstance(transform, list) or len(transform) != 4 or any(
        not isinstance(row, list) or len(row) != 4 for row in transform
    ):
        raise ValueError("T_map_from_motive must be a 4x4 matrix")
    try:
        transform = [[float(element) for element in row] for row in transform]
    except (TypeError, ValueError) as error:
        raise ValueError("T_map_from_motive must be numeric") from error
    if not all(math.isfinite(element) for row in transform for element in row):
        raise ValueError("T_map_from_motive contains a non-finite value")
    if any(abs(observed - expected) > 1e-8 for observed, expected in zip(transform[3], (0, 0, 0, 1))):
        raise ValueError("T_map_from_motive has an invalid homogeneous row")
    rotation = [row[:3] for row in transform[:3]]
    determinant = _determinant3(rotation)
    if abs(determinant - 1.0) > 1e-6:
        raise ValueError("Motive-to-map rotation is not proper")
    gram = [
        [sum(rotation[k][i] * rotation[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]
    if max(abs(gram[i][j] - (1.0 if i == j else 0.0)) for i in range(3) for j in range(3)) > 1e-6:
        raise ValueError("Motive-to-map rotation is not orthonormal")
    if abs(rotation[0][1]) > 1e-6 or abs(rotation[1][1]) > 1e-6 or abs(rotation[2][1] - 1.0) > 1e-6:
        raise ValueError("Motive Y-up axis does not map to positive Polycam Z-up")

    pelvis = value.get("pelvis_heading", {})
    if int(pelvis.get("rigid_body_id", -1)) != int(expected_pelvis_rigid_body_id):
        raise ValueError("Motive pelvis rigid-body ID mismatch")
    if pelvis.get("rigid_body_name") != expected_pelvis_rigid_body_name:
        raise ValueError("Motive pelvis rigid-body name mismatch")
    if pelvis.get("forward_axis") != "+x":
        raise ValueError("Motive pelvis heading must use the calibrated +x asset axis")
    try:
        yaw_offset = float(pelvis["yaw_offset_rad"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Motive pelvis yaw offset is missing or invalid") from error
    if not math.isfinite(yaw_offset):
        raise ValueError("Motive pelvis yaw offset is not finite")

    metrics = value.get("metrics", {})
    if int(metrics.get("inlier_count", 0)) < 4 or int(metrics.get("holdout_count", 0)) < 1:
        raise ValueError("Motive transform lacks the required fit and holdout evidence")
    return {
        "schema": "g1_motive_polycam_transform_validation_v1",
        "status": "pass",
        "content_sha256": observed_digest,
        "map_identity": expected_identity,
        "pelvis_rigid_body_id": int(expected_pelvis_rigid_body_id),
        "pelvis_rigid_body_name": expected_pelvis_rigid_body_name,
        "yaw_offset_rad": yaw_offset,
        "rotation_determinant": determinant,
        "command_capability": "structurally_unavailable",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--glb-sha256", required=True)
    parser.add_argument("--surface-sha256", required=True)
    parser.add_argument("--structural-map-sha256", required=True)
    parser.add_argument("--pelvis-rigid-body-id", type=int, required=True)
    parser.add_argument("--pelvis-rigid-body-name", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = json.loads(args.transform.read_text(encoding="utf-8"))
    try:
        result = validate_transform(
            value,
            expected_glb_sha256=args.glb_sha256,
            expected_surface_sha256=args.surface_sha256,
            expected_structural_map_sha256=args.structural_map_sha256,
            expected_pelvis_rigid_body_id=args.pelvis_rigid_body_id,
            expected_pelvis_rigid_body_name=args.pelvis_rigid_body_name,
        )
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
