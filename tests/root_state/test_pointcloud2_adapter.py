from dataclasses import dataclass
import struct

import numpy as np
import pytest

from g1_root_state_bridge.pointcloud2_adapter import (
    PointCloud2ContractError,
    decode_livox_pointcloud2,
)


@dataclass
class Field:
    name: str
    offset: int
    datatype: int = 7
    count: int = 1


class Message:
    height = 1
    width = 3
    point_step = 16
    row_step = 48
    is_bigendian = False
    fields = [Field("x", 0), Field("y", 4), Field("z", 8), Field("time", 12)]
    data = b"".join(
        struct.pack("<4f", *row)
        for row in (
            (1.0, 0.0, 0.0, 0.0),
            (20.0, 0.0, 0.0, 50_000_000.0),
            (2.0, 0.0, 0.0, 100_000_000.0),
        )
    )


def test_deployed_pointcloud_layout_preserves_time_and_range_filters() -> None:
    xyz, relative = decode_livox_pointcloud2(Message())
    np.testing.assert_allclose(xyz[:, 0], (1.0, 2.0))
    np.testing.assert_allclose(relative, (0.0, 0.1))


def test_normalized_replay_seconds_are_an_explicit_compatibility_mode() -> None:
    message = Message()
    message.data = b"".join(
        struct.pack("<4f", *row)
        for row in ((1.0, 0.0, 0.0, 0.0), (2.0, 0.0, 0.0, 0.1))
    )
    message.width = 2
    message.row_step = 32
    xyz, relative = decode_livox_pointcloud2(message, time_unit="seconds")
    np.testing.assert_allclose(xyz[:, 0], (1.0, 2.0))
    np.testing.assert_allclose(relative, (0.0, 0.1))


def test_wrong_time_unit_fails_closed_on_implausible_scan_duration() -> None:
    with pytest.raises(PointCloud2ContractError, match="scan duration"):
        decode_livox_pointcloud2(Message(), time_unit="seconds")


def test_missing_time_field_is_rejected() -> None:
    message = Message()
    message.fields = message.fields[:3]
    with pytest.raises(PointCloud2ContractError, match="missing fields: time"):
        decode_livox_pointcloud2(message)
