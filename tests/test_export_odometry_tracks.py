from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/export_odometry_tracks.py"


def load_module():
    spec = importlib.util.spec_from_file_location("export_odometry_tracks", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vector(x: float, y: float, z: float) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def test_odometry_record_preserves_frame_source_and_receipt_semantics() -> None:
    module = load_module()
    message = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=12, nanosec=345), frame_id="camera_init"
        ),
        child_frame_id="pelvis",
        pose=SimpleNamespace(
            pose=SimpleNamespace(
                position=vector(1.0, 2.0, 3.0),
                orientation=SimpleNamespace(x=0.1, y=0.2, z=0.3, w=0.9),
            )
        ),
        twist=SimpleNamespace(
            twist=SimpleNamespace(
                linear=vector(0.4, 0.5, 0.6), angular=vector(0.7, 0.8, 0.9)
            )
        ),
    )

    record = module.odometry_record(
        "/pelvis_state_estimation", message, receipt_time_ns=12_100_000_000
    )

    assert record == {
        "schema": "g1_odometry_track_v1",
        "kind": "odometry",
        "topic": "/pelvis_state_estimation",
        "source_time_ns": 12_000_000_345,
        "receipt_time_ns": 12_100_000_000,
        "frame_id": "camera_init",
        "child_frame_id": "pelvis",
        "position_xyz_m": [1.0, 2.0, 3.0],
        "quaternion_xyzw": [0.1, 0.2, 0.3, 0.9],
        "linear_velocity_xyz_mps": [0.4, 0.5, 0.6],
        "angular_velocity_xyz_radps": [0.7, 0.8, 0.9],
    }
