from array import array
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from scripts.normalize_unitree_pointcloud2_bag import (
    fit_header_clock,
    map_header_ns,
    normalize_time_field,
    scale_imu_acceleration,
)


@dataclass
class Field:
    name: str
    offset: int
    datatype: int
    count: int


def test_normalize_time_field_preserves_padding_and_scales_nanoseconds() -> None:
    width, height, point_step, row_step = 3, 2, 22, 70
    source = bytearray([0xA5] * (row_step * height))
    expected_ns = []
    for row in range(height):
        values = np.ndarray(
            (width,),
            dtype="<f4",
            buffer=source,
            offset=row * row_step + 18,
            strides=(point_step,),
        )
        values[:] = [5_000.0, 25_000_000.0, 50_000_000.0]
        expected_ns.extend(values.tolist())

    output, stats = normalize_time_field(
        array("B", source),
        fields=[Field("time", 18, 7, 1)],
        width=width,
        height=height,
        point_step=point_step,
        row_step=row_step,
        is_bigendian=False,
        scale=1e-9,
    )

    result = bytearray(output)
    actual_seconds = []
    for row in range(height):
        values = np.ndarray(
            (width,),
            dtype="<f4",
            buffer=result,
            offset=row * row_step + 18,
            strides=(point_step,),
        )
        actual_seconds.extend(values.tolist())
        assert result[row * row_step + width * point_step : (row + 1) * row_step] == bytes(
            [0xA5] * (row_step - width * point_step)
        )

    np.testing.assert_allclose(actual_seconds, np.asarray(expected_ns) * 1e-9)
    assert stats["points"] == 6
    assert stats["raw_time_min"] == 5_000.0
    assert stats["raw_time_max"] == 50_000_000.0
    assert stats["scaled_time_max_s"] == 0.05


def test_header_clock_fit_recovers_epoch_and_rate_without_receipt_jitter() -> None:
    source = 1_700_000_000_000_000_000 + np.arange(1000, dtype=np.int64) * 5_000_000
    true_target = 1_800_000_000_000_000_000 + np.rint(
        (source - source[0]) * 1.00002
    ).astype(np.int64)
    delay_ns = np.tile(np.asarray([1_000_000, 2_000_000, 8_000_000, 3_000_000]), 250)
    receipt = true_target + delay_ns
    mapping = fit_header_clock(source, receipt, lower_quantile=0.01)
    mapped = np.asarray([map_header_ns(value, mapping) for value in source])

    np.testing.assert_allclose(mapped, true_target + 1_000_000, atol=100_000)
    assert abs(float(mapping["scale"]) - 1.00002) < 5e-6
    assert float(mapping["delay_p95_ms"]) >= 6.9


def test_scale_imu_acceleration_changes_no_other_vector() -> None:
    message = SimpleNamespace(
        linear_acceleration=SimpleNamespace(x=0.1, y=-0.2, z=-1.0),
        angular_velocity=SimpleNamespace(x=1.0, y=2.0, z=3.0),
    )
    scale_imu_acceleration(message, 9.80665)
    np.testing.assert_allclose(
        [
            message.linear_acceleration.x,
            message.linear_acceleration.y,
            message.linear_acceleration.z,
        ],
        [0.980665, -1.96133, -9.80665],
    )
    assert (message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z) == (
        1.0,
        2.0,
        3.0,
    )


def test_scale_imu_acceleration_rejects_invalid_scale() -> None:
    message = SimpleNamespace(linear_acceleration=SimpleNamespace(x=0.0, y=0.0, z=0.0))
    for scale in (0.0, -1.0, float("nan")):
        try:
            scale_imu_acceleration(message, scale)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid IMU scale was accepted: {scale}")
