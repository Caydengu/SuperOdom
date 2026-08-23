from types import SimpleNamespace

import numpy as np

from scripts.export_livox_scans import bounded_pointcloud2, bounded_pointcloud2_with_time


def test_bounded_pointcloud2_filters_and_deterministically_caps() -> None:
    xyzt = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.00],
            [1.0, 0.0, 0.0, 0.01],
            [2.0, 0.0, 0.0, 0.02],
            [3.0, 0.0, 0.0, 0.03],
            [20.0, 0.0, 0.0, 0.04],
        ],
        dtype="<f4",
    )
    message = SimpleNamespace(
        fields=[
            SimpleNamespace(name="x", offset=0, datatype=7, count=1),
            SimpleNamespace(name="y", offset=4, datatype=7, count=1),
            SimpleNamespace(name="z", offset=8, datatype=7, count=1),
            SimpleNamespace(name="time", offset=12, datatype=7, count=1),
        ],
        width=5,
        height=1,
        is_bigendian=False,
        point_step=16,
        row_step=80,
        data=xyzt.tobytes(),
    )
    result = bounded_pointcloud2(
        message,
        minimum_range_m=0.5,
        maximum_range_m=10.0,
        maximum_points=2,
    )
    np.testing.assert_allclose(result, [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    result_with_time, relative_time_s = bounded_pointcloud2_with_time(
        message,
        minimum_range_m=0.5,
        maximum_range_m=10.0,
        maximum_points=2,
    )
    np.testing.assert_allclose(result_with_time, result)
    np.testing.assert_allclose(relative_time_s, [0.01, 0.03])
