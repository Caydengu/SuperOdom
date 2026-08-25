from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from g1_root_state_bridge.structural_map_localization import (
    AutomaticMapCorrectionEngine,
    StructuralMapConfig,
    _occupancy,
    correlate_at_yaw,
    refine_icp,
    vertical_persistence_xy,
)


def test_occupancy_is_binary_when_input_cells_repeat() -> None:
    points = np.asarray(((1.0, 2.0), (1.0, 2.0), (1.1, 2.0)))
    grid, minimum = _occupancy(points, 0.1)
    assert float(np.sum(grid)) == 2.0
    np.testing.assert_allclose(minimum, (1.0, 2.0))


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
    refined = refine_icp(
        map_xy,
        query,
        np.asarray(coarse["rotation"]),
        np.asarray(coarse["translation_m"]),
    )
    assert np.linalg.norm(refined["translation_m"] - translation) < 1e-3
    assert np.max(np.abs(refined["rotation"] - rotation)) < 1e-3


def test_vertical_persistence_rejects_floor_and_keeps_wall_columns() -> None:
    floor = np.asarray(
        [
            (x, y, 0.0)
            for x in np.arange(0.0, 1.0, 0.1)
            for y in np.arange(0.0, 1.0, 0.1)
        ]
    )
    wall = np.asarray(
        [(2.0, y, z) for y in (0.0, 0.1) for z in np.arange(-0.5, 0.6, 0.1)]
    )
    result = vertical_persistence_xy(np.concatenate((floor, wall)), xy_resolution_m=0.1)
    assert result.shape[0] == 2
    np.testing.assert_allclose(result[:, 0], 2.0)


def test_vertical_persistence_matches_two_axis_unique_reference() -> None:
    rng = np.random.default_rng(19)
    points = rng.uniform((-2.0, -3.0, -1.0), (2.0, 3.0, 1.0), size=(20_000, 3))
    xy_cell = np.rint(points[:, :2] / 0.1).astype(np.int64)
    z_cell = np.rint(points[:, 2] / 0.1).astype(np.int64)
    unique_xy, inverse = np.unique(xy_cell, axis=0, return_inverse=True)
    z_min = np.full(unique_xy.shape[0], np.inf)
    z_max = np.full(unique_xy.shape[0], -np.inf)
    np.minimum.at(z_min, inverse, points[:, 2])
    np.maximum.at(z_max, inverse, points[:, 2])
    unique_xy_z = np.unique(np.column_stack((inverse, z_cell)), axis=0)
    height_bins = np.bincount(unique_xy_z[:, 0], minlength=unique_xy.shape[0])
    admitted = ((z_max - z_min) >= 0.4) & (height_bins >= 4)
    xy_sum = np.zeros((unique_xy.shape[0], 2), dtype=np.float64)
    np.add.at(xy_sum, inverse, points[:, :2])
    expected = xy_sum / np.bincount(inverse)[:, None]
    result = vertical_persistence_xy(points)
    np.testing.assert_allclose(result, expected[admitted])


def test_heading_update_preserves_current_map_position(tmp_path: Path) -> None:
    map_path = tmp_path / "map.npz"
    rng = np.random.default_rng(11)
    map_xy = rng.uniform((-3.0, -2.0), (3.0, 2.0), size=(3000, 2)).astype(np.float32)
    np.savez(map_path, map_xy_all_5cm=map_xy)
    engine = AutomaticMapCorrectionEngine(map_path=map_path)
    engine.bind_local_epoch(3)
    engine.rotation = np.eye(2)
    engine.translation_m = np.asarray((1.0, -2.0))
    engine.sequence = 1
    local_position = np.asarray((0.5, 0.25))
    old_position = local_position @ engine.rotation + engine.translation_m
    attempt = engine.track(
        map_xy.astype(np.float64) - np.asarray((1.0, -2.0)),
        local_position_xy_m=local_position,
        reference_time_ns=1_000_000_000,
        evidence_time_ns=2_000_000_000,
        application_time_ns=2_010_000_000,
    )
    assert attempt.accepted
    assert attempt.packet is not None
    new_position = (
        attempt.packet.map_T_local[:3, :3]
        @ np.asarray((local_position[0], local_position[1], 0.0))
        + attempt.packet.map_T_local[:3, 3]
    )[:2]
    np.testing.assert_allclose(new_position, old_position, atol=1e-6)
    assert np.asarray(attempt.report["candidate_map_T_local"]).shape == (4, 4)
    np.testing.assert_allclose(
        attempt.packet.map_T_local,
        np.asarray(attempt.report["applied_map_T_local"]),
    )
    np.testing.assert_allclose(
        attempt.report["applied_map_T_local"],
        attempt.packet.map_T_local,
        atol=1e-6,
    )


def test_global_localization_configuration_preserves_frozen_yaw_grid() -> None:
    config = StructuralMapConfig()
    assert config.correlation_resolution_m == 0.15
    assert config.minimum_overlap_fraction == 0.12
    assert config.minimum_peak_ratio == 1.03
