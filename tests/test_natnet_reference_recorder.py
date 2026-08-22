from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[1] / "scripts" / "record_natnet_reference.py"
SPEC = importlib.util.spec_from_file_location("record_natnet_reference", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frame_record_preserves_native_motive_contract() -> None:
    body = SimpleNamespace(
        id_num=17,
        pos=(1.0, 2.0, 3.0),
        rot=(0.1, 0.2, 0.3, 0.9),
        tracking_valid=True,
        error=0.002,
    )
    data = {
        "frame_number": 42,
        "timestamp": 12.5,
        "is_recording": True,
        "rigid_body_data": SimpleNamespace(rigid_body_list=[body]),
    }

    record = MODULE.frame_record(
        data,
        target_id=17,
        target_name="G1_PELVIS",
        receipt_realtime_ns=100,
        receipt_monotonic_ns=200,
    )

    assert record is not None
    assert record["position_xyz_m_motive_native"] == [1.0, 2.0, 3.0]
    assert record["quaternion_xyzw_motive_native"] == [0.1, 0.2, 0.3, 0.9]
    assert record["receipt_realtime_ns"] == 100
    assert record["receipt_monotonic_ns"] == 200
    assert record["tracking_valid"] is True


def test_frame_record_rejects_wrong_rigid_body() -> None:
    data = {
        "frame_number": 1,
        "timestamp": 0.0,
        "rigid_body_data": SimpleNamespace(
            rigid_body_list=[SimpleNamespace(id_num=9)]
        ),
    }
    assert MODULE.frame_record(
        data,
        target_id=17,
        target_name="G1_PELVIS",
        receipt_realtime_ns=1,
        receipt_monotonic_ns=2,
    ) is None
