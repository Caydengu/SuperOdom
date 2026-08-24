"""ROS-independent decoding of the deployed Livox PointCloud2 layout."""

from __future__ import annotations

from typing import Any

import numpy as np


class PointCloud2ContractError(ValueError):
    """A PointCloud2 payload does not match the deployed Livox contract."""


def decode_xyz_pointcloud2(
    message: Any,
    *,
    maximum_points: int | None = None,
) -> np.ndarray:
    """Decode a finite XYZ-only PointCloud2 without ROS helper imports."""

    if maximum_points is not None and maximum_points <= 0:
        raise PointCloud2ContractError("maximum_points must be positive when set")
    fields = {str(field.name): field for field in message.fields}
    missing = sorted({"x", "y", "z"} - fields.keys())
    if missing:
        raise PointCloud2ContractError(
            f"PointCloud2 is missing fields: {', '.join(missing)}"
        )
    endian = ">" if bool(message.is_bigendian) else "<"
    columns = []
    for name in ("x", "y", "z"):
        field = fields[name]
        if int(field.datatype) != 7 or int(field.count) != 1:
            raise PointCloud2ContractError(
                f"PointCloud2 {name} must be one float32 field"
            )
        columns.append(
            np.ndarray(
                (int(message.height), int(message.width)),
                dtype=np.dtype(endian + "f4"),
                buffer=message.data,
                offset=int(field.offset),
                strides=(int(message.row_step), int(message.point_step)),
            ).reshape(-1)
        )
    xyz = np.column_stack(columns).astype(np.float32, copy=True)
    indices = np.flatnonzero(np.all(np.isfinite(xyz), axis=1))
    if maximum_points is not None and indices.size > maximum_points:
        indices = indices[
            np.linspace(0, indices.size - 1, maximum_points, dtype=np.int64)
        ]
    if indices.size == 0:
        raise PointCloud2ContractError("PointCloud2 contains no finite XYZ points")
    return xyz[indices]


def decode_livox_pointcloud2(
    message: Any,
    *,
    minimum_range_m: float = 0.5,
    maximum_range_m: float = 15.0,
    maximum_points: int = 5_000,
    time_unit: str = "nanoseconds",
    maximum_relative_time_s: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bounded xyz and relative seconds from the native G1 layout."""

    if (
        not 0.0 <= minimum_range_m < maximum_range_m
        or maximum_points <= 0
        or maximum_relative_time_s <= 0.0
    ):
        raise PointCloud2ContractError("range and point limits are invalid")
    if time_unit not in {"nanoseconds", "seconds"}:
        raise PointCloud2ContractError(
            "PointCloud2 time unit must be nanoseconds or seconds"
        )
    fields = {str(field.name): field for field in message.fields}
    missing = sorted({"x", "y", "z", "time"} - fields.keys())
    if missing:
        raise PointCloud2ContractError(
            f"PointCloud2 is missing fields: {', '.join(missing)}"
        )
    endian = ">" if bool(message.is_bigendian) else "<"
    columns = []
    for name in ("x", "y", "z", "time"):
        field = fields[name]
        if int(field.datatype) != 7 or int(field.count) != 1:
            raise PointCloud2ContractError(
                f"PointCloud2 {name} must be one float32 field"
            )
        columns.append(
            np.ndarray(
                (int(message.height), int(message.width)),
                dtype=np.dtype(endian + "f4"),
                buffer=message.data,
                offset=int(field.offset),
                strides=(int(message.row_step), int(message.point_step)),
            ).reshape(-1)
        )
    xyz = np.column_stack(columns[:3]).astype(np.float32, copy=True)
    time_scale = 1e-9 if time_unit == "nanoseconds" else 1.0
    relative_time_s = np.asarray(columns[3], dtype=np.float32).copy() * time_scale
    finite = np.all(np.isfinite(xyz), axis=1) & np.isfinite(relative_time_s)
    squared_range = np.sum(xyz.astype(np.float64) ** 2, axis=1)
    spatially_admitted = (
        finite
        & (squared_range >= minimum_range_m**2)
        & (squared_range <= maximum_range_m**2)
    )
    if np.any(
        spatially_admitted
        & ((relative_time_s < 0.0) | (relative_time_s > maximum_relative_time_s))
    ):
        raise PointCloud2ContractError(
            "PointCloud2 relative time exceeds the configured scan duration"
        )
    indices = np.flatnonzero(
        spatially_admitted
        & (relative_time_s >= 0.0)
        & (relative_time_s <= maximum_relative_time_s)
    )
    if indices.size > maximum_points:
        indices = indices[
            np.linspace(0, indices.size - 1, maximum_points, dtype=np.int64)
        ]
    if indices.size == 0:
        raise PointCloud2ContractError("PointCloud2 contains no admitted points")
    return xyz[indices], relative_time_s[indices]


def header_time_ns(message: Any) -> int:
    value = int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )
    if value <= 0:
        raise PointCloud2ContractError("message header timestamp must be positive")
    return value
