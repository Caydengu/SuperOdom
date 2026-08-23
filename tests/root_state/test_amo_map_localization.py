from __future__ import annotations

import math

import numpy as np
from g1_root_state_bridge.amo_map_localization import (
    correlate_at_yaw,
    directional_observability,
    refine_icp,
    vertical_persistence_xy,
)


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


def test_directional_observability_rejects_single_wall_tangent_ambiguity() -> None:
    wall = np.column_stack((np.linspace(-5.0, 5.0, 500), np.zeros(500)))
    target = np.arange(wall.shape[0], dtype=np.int64)
    result = directional_observability(
        wall,
        wall.copy(),
        target,
        np.ones(wall.shape[0], dtype=bool),
    )
    assert result["minimum_eigenvalue"] < 1e-6
    assert not result["observable"]


def test_vertical_persistence_rejects_floor_and_keeps_wall_columns() -> None:
    floor = np.asarray(
        [(x, y, 0.0) for x in np.arange(0.0, 1.0, 0.1) for y in np.arange(0.0, 1.0, 0.1)]
    )
    wall = np.asarray(
        [(2.0, y, z) for y in (0.0, 0.1) for z in np.arange(-0.5, 0.6, 0.1)]
    )
    result = vertical_persistence_xy(np.concatenate((floor, wall)), xy_resolution_m=0.1)
    assert result.shape[0] == 2
    np.testing.assert_allclose(result[:, 0], 2.0)
