from __future__ import annotations

import math

import numpy as np
from g1_root_state_bridge.amo_map_localization import correlate_at_yaw, refine_icp


def test_correlation_and_icp_recover_known_planar_transform() -> None:
    rng = np.random.default_rng(7)
    query = rng.uniform((-2.0, -1.0), (3.0, 2.0), size=(2000, 2))
    angle = math.radians(25.0)
    rotation = np.asarray(
        ((math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle)))
    )
    translation = np.asarray((4.2, -3.1))
    map_xy = query @ rotation + translation
    coarse = correlate_at_yaw(map_xy, query, angle_rad=angle, resolution_m=0.10)
    assert np.linalg.norm(coarse["translation_m"] - translation) < 0.15
    refined = refine_icp(
        map_xy,
        query,
        np.asarray(coarse["rotation"]),
        np.asarray(coarse["translation_m"]),
    )
    assert np.linalg.norm(refined["translation_m"] - translation) < 1e-3
    assert np.max(np.abs(refined["rotation"] - rotation)) < 1e-3
