from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from g1_root_state_bridge.structural_map_localization import (
    AutomaticMapCorrectionEngine,
    StructuralMapLocalizationError,
)
from g1_root_state_bridge.ui_initialization import (
    UIInitializationError,
    load_ui_map_initialization,
)


def _transform(yaw=0.3, x=1.2, y=-0.4):
    c, s = math.cos(yaw), math.sin(yaw)
    value = np.eye(4)
    value[:2, :2] = ((c, -s), (s, c))
    value[:2, 3] = (x, y)
    return value


def _write_receipt(path, structural_sha256, *, fitness=0.72):
    value = {
        "schema": "g1_map_ui_initialization_v1",
        "status": "accepted",
        "created_realtime_ns": 3_100_000_000,
        "command_capability": "structurally_unavailable",
        "map": {
            "glb_path": "/tmp/fieldbay.glb",
            "glb_sha256": "a" * 64,
            "surface_sha256": "b" * 64,
            "structural_map_sha256": structural_sha256,
            "sample_count": 500_000,
            "sample_seed": 24,
        },
        "scan": {
            "frame_id": "kiss_local",
            "window_start_ns": 1_000_000_000,
            "window_end_ns": 3_000_000_000,
            "evidence_time_ns": 3_000_000_000,
            "point_count": 40_000,
            "source": "test",
        },
        "pin": {
            "map_position_xyz_m": [1.0, -0.5, 1.2],
            "map_yaw_rad": 0.25,
            "T_map_local_seed": np.eye(4).tolist(),
        },
        "result": {
            "T_map_local": _transform().tolist(),
            "correction_m": 0.22,
            "correction_yaw_deg": 2.0,
            "localizer_status": "OK",
            "fitness": fitness,
            "inlier_frac": fitness,
            "rmse_m": 0.08,
            "min_eig": 0.05,
            "cond_number": 120.0,
            "n_iter": 9,
            "snap_history": [],
            "maximum_correction_m": 0.75,
            "maximum_correction_yaw_deg": 20.0,
            "maximum_rmse_m": 0.12,
            "minimum_fitness": 0.45,
            "minimum_eigenvalue": 0.02,
            "maximum_condition_number": 1e6,
            "rejection_reason": None,
        },
    }
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    value["content_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_ui_receipt_binds_all_map_identities(tmp_path):
    path = tmp_path / "initialization.json"
    _write_receipt(path, "c" * 64)
    value = load_ui_map_initialization(
        path,
        expected_structural_map_sha256="c" * 64,
        expected_glb_sha256="a" * 64,
        expected_surface_sha256="b" * 64,
    )
    assert value.frame_id == "kiss_local"
    assert value.created_realtime_ns == 3_100_000_000
    assert value.fitness == pytest.approx(0.72)
    assert value.map_T_local == pytest.approx(_transform())


def test_ui_receipt_tamper_and_loose_writer_gate_fail_closed(tmp_path):
    path = tmp_path / "initialization.json"
    _write_receipt(path, "c" * 64)
    value = json.loads(path.read_text())
    value["result"]["maximum_rmse_m"] = 0.2
    canonical = json.dumps(
        {k: v for k, v in value.items() if k != "content_sha256"},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    value["content_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(UIInitializationError, match="writer RMSE gate"):
        load_ui_map_initialization(
            path,
            expected_structural_map_sha256="c" * 64,
            expected_glb_sha256="a" * 64,
            expected_surface_sha256="b" * 64,
        )
    value["result"]["maximum_rmse_m"] = 0.12
    value["content_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(UIInitializationError, match="content digest"):
        load_ui_map_initialization(
            path,
            expected_structural_map_sha256="c" * 64,
            expected_glb_sha256="a" * 64,
            expected_surface_sha256="b" * 64,
        )


def test_engine_emits_ui_transform_as_first_map_packet(tmp_path):
    map_path = tmp_path / "map.npz"
    np.savez_compressed(
        map_path,
        map_xy_structural_2cm=np.asarray(
            ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
        ),
    )
    engine = AutomaticMapCorrectionEngine(
        map_path=map_path, map_key="map_xy_structural_2cm"
    )
    engine.bind_local_epoch(7)
    attempt = engine.initialize_from_ui(
        map_T_local=_transform(),
        fitness=0.72,
        rmse_m=0.08,
        min_eig=0.05,
        cond_number=120.0,
        reference_time_ns=1_000_000_000,
        evidence_time_ns=3_000_000_000,
        application_time_ns=10_000_000_000,
    )
    assert attempt.accepted
    assert attempt.packet is not None
    assert attempt.packet.local_source_epoch == 7
    assert attempt.packet.map_T_local == pytest.approx(_transform())
    with pytest.raises(StructuralMapLocalizationError, match="stale"):
        engine.bind_local_epoch(8)
        engine.initialize_from_ui(
            map_T_local=_transform(),
            fitness=0.72,
            rmse_m=0.08,
            min_eig=0.05,
            cond_number=120.0,
            reference_time_ns=1_000_000_000,
            evidence_time_ns=3_000_000_000,
            application_time_ns=33_000_000_001,
        )
    with pytest.raises(StructuralMapLocalizationError, match="fitness gate"):
        engine.bind_local_epoch(9)
        engine.initialize_from_ui(
            map_T_local=_transform(),
            fitness=0.2,
            rmse_m=0.08,
            min_eig=0.05,
            cond_number=120.0,
            reference_time_ns=4_000_000_000,
            evidence_time_ns=5_000_000_000,
            application_time_ns=5_100_000_000,
        )
