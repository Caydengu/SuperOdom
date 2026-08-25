from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.motive_polycam_calibration_app import MotiveMarkerBuffer


def _marker_set(name: str, points: list[tuple[float, float, float]]):
    return SimpleNamespace(model_name=name.encode(), marker_pos_list=points)


def _body(*, valid: bool = True, error: float = 0.0007):
    return SimpleNamespace(tracking_valid=valid, error=error)


def test_fixed_anchor_buffer_exposes_stable_marker_set_points() -> None:
    buffer = MotiveMarkerBuffer({43: "STAIRS_TOP", 44: "SHELF_MIDDLE_EDGE"})
    now_ns = time.monotonic_ns()
    stair = _marker_set("STAIRS_TOP", [(1.0, 0.5, 2.0), (2.0, 0.5, 2.0)])
    shelf = _marker_set("SHELF_MIDDLE_EDGE", [(4.0, 1.4, 7.0)])
    for _ in range(100):
        buffer.append([stair, shelf], {43: _body(), 44: _body()}, now_ns)

    sources = buffer.sources()
    assert sources["STAIRS_TOP [43] Marker 001"] == (43, 0)
    assert sources["STAIRS_TOP [43] Marker 002"] == (43, 1)
    assert sources["SHELF_MIDDLE_EDGE [44] Marker 001"] == (44, 0)
    position, evidence = buffer.stable_position(
        (43, 1), window_sec=1.5, max_std_m=0.004
    )
    np.testing.assert_allclose(position, (2.0, 0.5, 2.0))
    assert evidence["rigid_body_id"] == 43
    assert evidence["marker_id"] == 1
    assert evidence["sample_count"] == 100


def test_fixed_anchor_buffer_rejects_untracked_or_moving_points() -> None:
    buffer = MotiveMarkerBuffer({43: "STAIRS_TOP"})
    now_ns = time.monotonic_ns()
    marker_set = _marker_set("STAIRS_TOP", [(1.0, 0.5, 2.0)])
    for _ in range(100):
        buffer.append([marker_set], {43: _body(valid=False)}, now_ns)
    assert buffer.sources() == {}

    for index in range(100):
        moving = _marker_set("STAIRS_TOP", [(1.0 + 0.02 * (index % 2), 0.5, 2.0)])
        buffer.append([moving], {43: _body()}, now_ns)
    with pytest.raises(ValueError, match="fixed marker is moving"):
        buffer.stable_position((43, 0), window_sec=1.5, max_std_m=0.004)
