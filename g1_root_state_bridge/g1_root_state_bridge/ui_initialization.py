"""Fail-closed reader for Gio UI's map-initialization receipt.

The file is an offboard operator-to-localizer seam. It contains no Motive data
and grants no command capability. The map node independently rechecks all
identities and the exact robot-vlm map gates before emitting RVMAP001.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

UI_INITIALIZATION_SCHEMA = "g1_map_ui_initialization_v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")


class UIInitializationError(ValueError):
    """The UI receipt cannot initialize the production map lane."""


def _sha(value: Any, name: str) -> str:
    text = str(value)
    if _SHA256.fullmatch(text) is None:
        raise UIInitializationError(f"{name} must be a lowercase SHA-256")
    return text


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise UIInitializationError(f"{name} must be finite")
    return result


def _transform(value: Any) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise UIInitializationError("result.T_map_local must be a finite 4x4 transform")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise UIInitializationError("result.T_map_local must be homogeneous")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4):
        raise UIInitializationError("result.T_map_local rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise UIInitializationError("result.T_map_local rotation must be proper")
    # The production consumer is planar. Reject a tilted map transform rather
    # than silently discarding roll/pitch from a bad 3D fit.
    if not np.allclose(rotation[2], (0.0, 0.0, 1.0), atol=2e-2) or not np.allclose(
        rotation[:, 2], (0.0, 0.0, 1.0), atol=2e-2
    ):
        raise UIInitializationError("result.T_map_local must be gravity aligned")
    return transform


@dataclass(frozen=True)
class UIMapInitialization:
    content_sha256: str
    glb_sha256: str
    surface_sha256: str
    structural_map_sha256: str
    frame_id: str
    reference_time_ns: int
    evidence_time_ns: int
    map_T_local: np.ndarray
    fitness: float
    rmse_m: float
    min_eig: float
    cond_number: float
    correction_m: float
    correction_yaw_deg: float
    point_count: int


def load_ui_map_initialization(
    path: str | Path,
    *,
    expected_structural_map_sha256: str,
    expected_glb_sha256: str | None = None,
    expected_surface_sha256: str | None = None,
    minimum_fitness: float = 0.45,
    maximum_rmse_m: float = 0.12,
    minimum_eigenvalue: float = 0.02,
    maximum_condition_number: float = 1e6,
    maximum_correction_m: float = 0.75,
    maximum_correction_yaw_deg: float = 20.0,
) -> UIMapInitialization:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    recorded_id = value.pop("content_sha256", None)
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    content_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if recorded_id != content_sha256:
        raise UIInitializationError("receipt content digest mismatch")
    if value.get("schema") != UI_INITIALIZATION_SCHEMA:
        raise UIInitializationError("receipt schema mismatch")
    if value.get("status") != "accepted":
        raise UIInitializationError("receipt was not accepted by the UI")
    if value.get("command_capability") != "structurally_unavailable":
        raise UIInitializationError("receipt must be command-incapable")

    map_row = value.get("map", {})
    scan = value.get("scan", {})
    result = value.get("result", {})
    glb_sha256 = _sha(map_row.get("glb_sha256", ""), "map.glb_sha256")
    surface_sha256 = _sha(map_row.get("surface_sha256", ""), "map.surface_sha256")
    structural_sha256 = _sha(
        map_row.get("structural_map_sha256", ""), "map.structural_map_sha256"
    )
    if structural_sha256 != _sha(
        expected_structural_map_sha256, "expected_structural_map_sha256"
    ):
        raise UIInitializationError("structural map digest mismatch")
    if expected_glb_sha256 is not None and glb_sha256 != _sha(
        expected_glb_sha256, "expected_glb_sha256"
    ):
        raise UIInitializationError("GLB digest mismatch")
    if expected_surface_sha256 is not None and surface_sha256 != _sha(
        expected_surface_sha256, "expected_surface_sha256"
    ):
        raise UIInitializationError("surface target digest mismatch")

    frame_id = str(scan.get("frame_id", ""))
    if frame_id != "kiss_local":
        raise UIInitializationError("receipt frame must be kiss_local")
    reference_ns = int(scan.get("window_start_ns", 0))
    window_end_ns = int(scan.get("window_end_ns", 0))
    evidence_ns = int(scan.get("evidence_time_ns", 0))
    if min(reference_ns, window_end_ns, evidence_ns) <= 0 or not (
        reference_ns <= window_end_ns <= evidence_ns
    ):
        raise UIInitializationError("receipt evidence times are invalid")
    point_count = int(scan.get("point_count", 0))
    if point_count < 300:
        raise UIInitializationError("receipt contains too few map points")

    transform = _transform(result.get("T_map_local"))
    fitness = _finite(result.get("fitness"), "result.fitness")
    rmse_m = _finite(result.get("rmse_m"), "result.rmse_m")
    min_eig = _finite(result.get("min_eig"), "result.min_eig")
    cond_number = _finite(result.get("cond_number"), "result.cond_number")
    correction_m = _finite(result.get("correction_m"), "result.correction_m")
    correction_yaw_deg = _finite(
        result.get("correction_yaw_deg"), "result.correction_yaw_deg"
    )
    if result.get("localizer_status") != "OK":
        raise UIInitializationError("localizer status is not OK")
    if fitness < minimum_fitness:
        raise UIInitializationError("fitness gate")
    if rmse_m >= maximum_rmse_m:
        raise UIInitializationError("RMSE gate")
    if min_eig < minimum_eigenvalue:
        raise UIInitializationError("observability gate")
    if cond_number > maximum_condition_number:
        raise UIInitializationError("condition-number gate")
    if correction_m > maximum_correction_m:
        raise UIInitializationError("pin translation gate")
    if correction_yaw_deg > maximum_correction_yaw_deg:
        raise UIInitializationError("pin yaw gate")

    # The writer records the thresholds it applied. It may be stricter, but it
    # may not claim a looser contract than this consumer independently applies.
    if float(result.get("minimum_fitness", 0.0)) < minimum_fitness:
        raise UIInitializationError("writer fitness gate is too loose")
    if float(result.get("maximum_rmse_m", math.inf)) > maximum_rmse_m:
        raise UIInitializationError("writer RMSE gate is too loose")
    if float(result.get("minimum_eigenvalue", 0.0)) < minimum_eigenvalue:
        raise UIInitializationError("writer observability gate is too loose")
    if float(result.get("maximum_condition_number", math.inf)) > maximum_condition_number:
        raise UIInitializationError("writer condition-number gate is too loose")
    if float(result.get("maximum_correction_m", math.inf)) > maximum_correction_m:
        raise UIInitializationError("writer translation gate is too loose")
    if float(result.get("maximum_correction_yaw_deg", math.inf)) > maximum_correction_yaw_deg:
        raise UIInitializationError("writer yaw gate is too loose")

    return UIMapInitialization(
        content_sha256=content_sha256,
        glb_sha256=glb_sha256,
        surface_sha256=surface_sha256,
        structural_map_sha256=structural_sha256,
        frame_id=frame_id,
        reference_time_ns=reference_ns,
        evidence_time_ns=evidence_ns,
        map_T_local=transform,
        fitness=fitness,
        rmse_m=rmse_m,
        min_eig=min_eig,
        cond_number=cond_number,
        correction_m=correction_m,
        correction_yaw_deg=correction_yaw_deg,
        point_count=point_count,
    )
