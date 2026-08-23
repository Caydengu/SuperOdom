#!/usr/bin/env python3
"""Export a deterministic, bounded Livox scan archive from a ROS 2 bag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def bounded_points(
    message: Any,
    *,
    minimum_range_m: float,
    maximum_range_m: float,
    maximum_points: int,
) -> np.ndarray:
    xyz = np.asarray(
        [(point.x, point.y, point.z) for point in message.points], dtype=np.float32
    )
    finite = np.all(np.isfinite(xyz), axis=1)
    squared_range = np.sum(xyz.astype(np.float64) ** 2, axis=1)
    admitted = (
        finite
        & (squared_range >= minimum_range_m**2)
        & (squared_range <= maximum_range_m**2)
    )
    xyz = xyz[admitted]
    if xyz.shape[0] > maximum_points:
        # Evenly spaced deterministic selection preserves the rotating Livox
        # pattern better than taking one contiguous prefix.
        indices = np.linspace(0, xyz.shape[0] - 1, maximum_points, dtype=np.int64)
        xyz = xyz[indices]
    return xyz


def bounded_points_with_time(
    message: Any,
    *,
    minimum_range_m: float,
    maximum_range_m: float,
    maximum_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound a Livox CustomMsg while preserving each point's acquisition time."""
    xyz = np.asarray(
        [(point.x, point.y, point.z) for point in message.points], dtype=np.float32
    )
    relative_time_s = np.asarray(
        [float(point.offset_time) * 1e-9 for point in message.points], dtype=np.float32
    )
    return _filter_and_cap(
        xyz,
        relative_time_s,
        minimum_range_m=minimum_range_m,
        maximum_range_m=maximum_range_m,
        maximum_points=maximum_points,
    )


def _filter_and_cap(
    xyz: np.ndarray,
    relative_time_s: np.ndarray,
    *,
    minimum_range_m: float,
    maximum_range_m: float,
    maximum_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if relative_time_s.shape != (xyz.shape[0],):
        raise ValueError("point-relative times must have one value per point")
    finite = np.all(np.isfinite(xyz), axis=1) & np.isfinite(relative_time_s)
    squared_range = np.sum(xyz.astype(np.float64) ** 2, axis=1)
    admitted_indices = np.flatnonzero(
        finite
        & (squared_range >= minimum_range_m**2)
        & (squared_range <= maximum_range_m**2)
    )
    if admitted_indices.size > maximum_points:
        selection = np.linspace(
            0, admitted_indices.size - 1, maximum_points, dtype=np.int64
        )
        admitted_indices = admitted_indices[selection]
    return xyz[admitted_indices], relative_time_s[admitted_indices]


def bounded_pointcloud2(
    message: Any,
    *,
    minimum_range_m: float,
    maximum_range_m: float,
    maximum_points: int,
) -> np.ndarray:
    xyz, _ = bounded_pointcloud2_with_time(
        message,
        minimum_range_m=minimum_range_m,
        maximum_range_m=maximum_range_m,
        maximum_points=maximum_points,
    )
    return xyz


def bounded_pointcloud2_with_time(
    message: Any,
    *,
    minimum_range_m: float,
    maximum_range_m: float,
    maximum_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    fields = {str(field.name): field for field in message.fields}
    missing = sorted({"x", "y", "z", "time"} - fields.keys())
    if missing:
        raise ValueError(f"PointCloud2 is missing fields: {', '.join(missing)}")
    endian = ">" if bool(message.is_bigendian) else "<"
    columns = []
    for name in ("x", "y", "z", "time"):
        field = fields[name]
        if int(field.datatype) != 7 or int(field.count) != 1:
            raise ValueError(f"PointCloud2 {name} must be one float32 field")
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
    relative_time_s = np.asarray(columns[3], dtype=np.float32).copy()
    return _filter_and_cap(
        xyz,
        relative_time_s,
        minimum_range_m=minimum_range_m,
        maximum_range_m=maximum_range_m,
        maximum_points=maximum_points,
    )


def main() -> None:
    import rosbag2_py
    from livox_ros_driver2.msg import CustomMsg
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu, PointCloud2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", default="/livox/lidar")
    parser.add_argument(
        "--imu-topic",
        default=None,
        help="Optional sensor_msgs/Imu topic to include for exact scan deskew.",
    )
    parser.add_argument("--minimum-range-m", type=float, default=0.5)
    parser.add_argument("--maximum-range-m", type=float, default=15.0)
    parser.add_argument("--maximum-points-per-scan", type=int, default=5000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    available_types = {
        topic.name: topic.type for topic in reader.get_all_topics_and_types()
    }
    message_type_name = available_types.get(args.topic)
    supported = {
        "livox_ros_driver2/msg/CustomMsg": CustomMsg,
        "sensor_msgs/msg/PointCloud2": PointCloud2,
    }
    if message_type_name not in supported:
        raise ValueError(
            f"{args.topic} must be Livox CustomMsg or PointCloud2; found {message_type_name!r}"
        )
    message_type = supported[message_type_name]
    if args.imu_topic is not None:
        imu_type_name = available_types.get(args.imu_topic)
        if imu_type_name != "sensor_msgs/msg/Imu":
            raise ValueError(
                f"{args.imu_topic} must be sensor_msgs/msg/Imu; found {imu_type_name!r}"
            )
    points: list[np.ndarray] = []
    relative_times: list[np.ndarray] = []
    offsets = [0]
    source_time_ns: list[int] = []
    receipt_time_ns: list[int] = []
    imu_source_time_ns: list[int] = []
    imu_receipt_time_ns: list[int] = []
    imu_angular_velocity_radps: list[tuple[float, float, float]] = []
    imu_linear_acceleration_mps2: list[tuple[float, float, float]] = []
    while reader.has_next():
        topic, serialized, receipt = reader.read_next()
        if args.imu_topic is not None and topic == args.imu_topic:
            imu = deserialize_message(serialized, Imu)
            imu_source_time_ns.append(
                int(imu.header.stamp.sec) * 1_000_000_000
                + int(imu.header.stamp.nanosec)
            )
            imu_receipt_time_ns.append(int(receipt))
            imu_angular_velocity_radps.append(
                (
                    float(imu.angular_velocity.x),
                    float(imu.angular_velocity.y),
                    float(imu.angular_velocity.z),
                )
            )
            imu_linear_acceleration_mps2.append(
                (
                    float(imu.linear_acceleration.x),
                    float(imu.linear_acceleration.y),
                    float(imu.linear_acceleration.z),
                )
            )
            continue
        if topic != args.topic:
            continue
        message = deserialize_message(serialized, message_type)
        bounded = (
            bounded_points_with_time
            if message_type_name == "livox_ros_driver2/msg/CustomMsg"
            else bounded_pointcloud2_with_time
        )
        scan, scan_time = bounded(
            message,
            minimum_range_m=args.minimum_range_m,
            maximum_range_m=args.maximum_range_m,
            maximum_points=args.maximum_points_per_scan,
        )
        points.append(scan)
        relative_times.append(scan_time)
        offsets.append(offsets[-1] + scan.shape[0])
        source_time_ns.append(
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        receipt_time_ns.append(int(receipt))
    if not points:
        raise ValueError(f"bag contains no {args.topic} messages")
    if args.imu_topic is not None and not imu_source_time_ns:
        raise ValueError(f"bag contains no {args.imu_topic} messages")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        points_xyz_m=np.concatenate(points, axis=0),
        point_relative_time_s=np.concatenate(relative_times, axis=0),
        scan_offsets=np.asarray(offsets, dtype=np.int64),
        source_time_ns=np.asarray(source_time_ns, dtype=np.int64),
        receipt_time_ns=np.asarray(receipt_time_ns, dtype=np.int64),
        imu_source_time_ns=np.asarray(imu_source_time_ns, dtype=np.int64),
        imu_receipt_time_ns=np.asarray(imu_receipt_time_ns, dtype=np.int64),
        imu_angular_velocity_radps=np.asarray(
            imu_angular_velocity_radps, dtype=np.float32
        ).reshape((-1, 3)),
        imu_linear_acceleration_mps2=np.asarray(
            imu_linear_acceleration_mps2, dtype=np.float32
        ).reshape((-1, 3)),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "schema": "g1_livox_scan_archive_v2",
                    "topic": args.topic,
                    "message_type": message_type_name,
                    "imu_topic": args.imu_topic,
                    "point_time_units": "seconds relative to scan header",
                    "minimum_range_m": args.minimum_range_m,
                    "maximum_range_m": args.maximum_range_m,
                    "maximum_points_per_scan": args.maximum_points_per_scan,
                    "selection": "range_filter_then_evenly_spaced_deterministic_cap",
                },
                sort_keys=True,
            )
        ),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scan_count": len(points),
                "point_count": offsets[-1],
                "imu_count": len(imu_source_time_ns),
                "duration_s": (source_time_ns[-1] - source_time_ns[0]) * 1e-9,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
