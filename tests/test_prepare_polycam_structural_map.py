import numpy as np

from scripts.prepare_polycam_structural_map import voxelize_xy


def test_voxelize_xy_is_deterministic_and_preserves_first_point() -> None:
    points = np.asarray([[0.001, 0.001], [0.009, 0.009], [0.031, 0.0]])
    actual = voxelize_xy(points, 0.02)
    np.testing.assert_allclose(actual, [[0.001, 0.001], [0.031, 0.0]])
