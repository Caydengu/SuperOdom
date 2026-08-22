from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/export_superodom_diagnostics.py"


def load_module():
    spec = importlib.util.spec_from_file_location("export_superodom_diagnostics", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vector(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def test_imu_record_keeps_capture_and_receipt_time_separate() -> None:
    module = load_module()
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=3, nanosec=4)),
        angular_velocity=vector(0.1, 0.2, 0.3),
        linear_acceleration=vector(1.0, 2.0, 3.0),
    )
    result = module.record("/livox/imu", message, 3_100_000_000)
    assert result["source_time_ns"] == 3_000_000_004
    assert result["receipt_time_ns"] == 3_100_000_000
    assert result["angular_velocity_xyz_radps"] == [0.1, 0.2, 0.3]
    assert result["linear_acceleration_xyz_mps2"] == [1.0, 2.0, 3.0]


def test_stats_record_preserves_observability_and_failure_counts() -> None:
    module = load_module()
    values = {
        "total_translation": 1.0,
        "total_rotation": 2.0,
        "translation_from_last": 0.1,
        "rotation_from_last": 0.2,
        "time_elapsed": 3.0,
        "latency": 4.0,
        "n_iterations": 5,
        "average_distance": 0.06,
        "uncertainty_x": 1.0,
        "uncertainty_y": 2.0,
        "uncertainty_z": 3.0,
        "uncertainty_roll": 4.0,
        "uncertainty_pitch": 5.0,
        "uncertainty_yaw": 6.0,
        "plane_match_success": 100,
        "plane_no_enough_neighbor": 1,
        "plane_neighbor_too_far": 2,
        "plane_badpca_structure": 3,
        "plane_invalid_numerical": 4,
        "plane_mse_too_large": 5,
        "plane_unknown": 6,
        "prediction_source": 7,
    }
    message = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(sec=8, nanosec=9)), **values
    )
    result = module.record("/super_odometry_stats", message, 8_100_000_000)
    assert result["translation_from_last_m"] == 0.1
    assert result["uncertainty_xyz_rpy"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert result["plane_match_success"] == 100
    assert result["plane_failure_counts"]["mse_too_large"] == 5
