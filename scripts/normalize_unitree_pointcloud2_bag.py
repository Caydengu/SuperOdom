#!/usr/bin/env python3
"""Normalize Unitree MID-360 replay timing and optional IMU acceleration units.

The G1 native projection stores ``time`` as float32 nanoseconds within each
scan. SuperOdometry's PointCloud2 ingestion path expects float32 seconds. This
tool rewrites only that field by default.  ``--imu-accel-scale`` additionally
supports a controlled replay treatment for G1 captures whose Livox IMU reports
acceleration in g even though ``sensor_msgs/Imu`` requires m/s^2.
"""

from __future__ import annotations

import argparse
import json
from array import array
from pathlib import Path
from typing import Any

import numpy as np


POINT_FIELD_FLOAT32 = 7


def scale_imu_acceleration(message: Any, scale: float) -> None:
    """Scale only the linear-acceleration vector of an Imu-like message."""
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"IMU acceleration scale must be positive and finite: {scale}")
    message.linear_acceleration.x *= float(scale)
    message.linear_acceleration.y *= float(scale)
    message.linear_acceleration.z *= float(scale)


def fit_header_clock(
    header_ns: np.ndarray,
    receipt_ns: np.ndarray,
    *,
    lower_quantile: float = 0.01,
) -> dict[str, float | int]:
    """Fit header time into the rosbag clock using its least-delayed envelope."""
    source = np.asarray(header_ns, dtype=np.int64)
    target = np.asarray(receipt_ns, dtype=np.int64)
    if source.ndim != 1 or target.shape != source.shape or source.size < 3:
        raise ValueError("header clock fit requires equal 1D arrays with >=3 samples")
    source_origin = int(source[0])
    target_origin = int(target[0])
    x = source.astype(np.float64) - source_origin
    y = target.astype(np.float64) - target_origin
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    scale = float(np.dot(xc, yc) / np.dot(xc, xc))
    if not 0.999 <= scale <= 1.001:
        raise ValueError(f"header clock scale is implausible: {scale}")
    intercept = float(np.mean(y) - scale * np.mean(x))
    residual = y - (intercept + scale * x)
    lower = float(np.quantile(residual, lower_quantile))
    delay = residual - lower
    return {
        "source_origin_ns": source_origin,
        "target_origin_ns": target_origin,
        "scale": scale,
        "lower_envelope_offset_ns": intercept + lower,
        "delay_p50_ms": float(np.quantile(delay, 0.50) * 1e-6),
        "delay_p95_ms": float(np.quantile(delay, 0.95) * 1e-6),
        "sample_count": int(source.size),
    }


def map_header_ns(header_ns: int, mapping: dict[str, float | int]) -> int:
    return int(
        round(
            int(mapping["target_origin_ns"])
            + float(mapping["lower_envelope_offset_ns"])
            + float(mapping["scale"])
            * (int(header_ns) - int(mapping["source_origin_ns"]))
        )
    )


def normalize_time_field(
    data: bytes | bytearray | array,
    *,
    fields: list[Any],
    width: int,
    height: int,
    point_step: int,
    row_step: int,
    is_bigendian: bool,
    scale: float,
) -> tuple[array, dict[str, float | int]]:
    """Return a copy with the float32 ``time`` field scaled in every point."""
    matches = [field for field in fields if str(field.name) == "time"]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one PointCloud2 time field, found {len(matches)}")
    field = matches[0]
    if int(field.datatype) != POINT_FIELD_FLOAT32 or int(field.count) != 1:
        raise ValueError(
            "PointCloud2 time must be one float32 value; "
            f"got datatype={field.datatype} count={field.count}"
        )
    offset = int(field.offset)
    if offset < 0 or offset + 4 > point_step:
        raise ValueError(f"time field offset {offset} is outside point_step {point_step}")
    expected = int(row_step) * int(height)
    if len(data) != expected:
        raise ValueError(f"PointCloud2 data has {len(data)} bytes, expected {expected}")

    output = bytearray(data)
    dtype = np.dtype(">f4" if is_bigendian else "<f4")
    minima: list[float] = []
    maxima: list[float] = []
    for row in range(int(height)):
        values = np.ndarray(
            shape=(int(width),),
            dtype=dtype,
            buffer=output,
            offset=row * int(row_step) + offset,
            strides=(int(point_step),),
        )
        if values.size:
            minima.append(float(np.min(values)))
            maxima.append(float(np.max(values)))
            values *= float(scale)
    raw_min = min(minima) if minima else float("nan")
    raw_max = max(maxima) if maxima else float("nan")
    return array("B", output), {
        "points": int(width) * int(height),
        "raw_time_min": raw_min,
        "raw_time_max": raw_max,
        "scaled_time_min_s": raw_min * float(scale),
        "scaled_time_max_s": raw_max * float(scale),
    }


def normalize_bag(
    input_bag: Path,
    output_bag: Path,
    *,
    lidar_topic: str,
    scale: float,
    align_header_to_bag_clock: bool,
    imu_topic: str,
    imu_accel_scale: float = 1.0,
) -> dict[str, object]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    from sensor_msgs.msg import Imu, PointCloud2

    if output_bag.exists():
        raise FileExistsError(f"refusing to overwrite output bag: {output_bag}")
    output_bag.parent.mkdir(parents=True, exist_ok=True)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(input_bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    topics = reader.get_all_topics_and_types()
    types = {topic.name: topic.type for topic in topics}
    expected_type = "sensor_msgs/msg/PointCloud2"
    if types.get(lidar_topic) != expected_type:
        raise ValueError(
            f"{lidar_topic} must have type {expected_type}; found {types.get(lidar_topic)!r}"
        )
    expected_imu_type = "sensor_msgs/msg/Imu"
    if types.get(imu_topic) != expected_imu_type:
        raise ValueError(
            f"{imu_topic} must have type {expected_imu_type}; found {types.get(imu_topic)!r}"
        )

    header_mappings: dict[str, dict[str, float | int]] = {}
    if align_header_to_bag_clock:
        samples: dict[str, tuple[list[int], list[int]]] = {
            lidar_topic: ([], []),
            imu_topic: ([], []),
        }
        while reader.has_next():
            topic, serialized, receipt_time_ns = reader.read_next()
            if topic not in samples:
                continue
            message_type = PointCloud2 if topic == lidar_topic else Imu
            message = deserialize_message(serialized, message_type)
            header_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
                message.header.stamp.nanosec
            )
            samples[topic][0].append(header_ns)
            samples[topic][1].append(int(receipt_time_ns))
        header_mappings = {
            topic: fit_header_clock(np.asarray(header), np.asarray(receipt))
            for topic, (header, receipt) in samples.items()
        }
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(input_bag), storage_id="sqlite3"),
            rosbag2_py.ConverterOptions("cdr", "cdr"),
        )

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output_bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    for topic in topics:
        writer.create_topic(topic)

    counts = {topic.name: 0 for topic in topics}
    point_count = 0
    raw_min = float("inf")
    raw_max = float("-inf")
    scaled_max = float("-inf")
    while reader.has_next():
        topic, serialized, receipt_time_ns = reader.read_next()
        if topic in (lidar_topic, imu_topic):
            message_type = PointCloud2 if topic == lidar_topic else Imu
            message = deserialize_message(serialized, message_type)
            if align_header_to_bag_clock:
                old_header_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(
                    message.header.stamp.nanosec
                )
                # LiDAR and IMU header stamps share the same robot clock. Apply
                # the high-rate, low-latency IMU mapping to both so the LiDAR's
                # ~scan-duration acquisition lag relative to receipt time is
                # preserved instead of incorrectly fitted away per topic.
                new_header_ns = map_header_ns(old_header_ns, header_mappings[imu_topic])
                message.header.stamp.sec = new_header_ns // 1_000_000_000
                message.header.stamp.nanosec = new_header_ns % 1_000_000_000
        if topic == lidar_topic:
            message.data, stats = normalize_time_field(
                message.data,
                fields=list(message.fields),
                width=message.width,
                height=message.height,
                point_step=message.point_step,
                row_step=message.row_step,
                is_bigendian=message.is_bigendian,
                scale=scale,
            )
            point_count += int(stats["points"])
            raw_min = min(raw_min, float(stats["raw_time_min"]))
            raw_max = max(raw_max, float(stats["raw_time_max"]))
            scaled_max = max(scaled_max, float(stats["scaled_time_max_s"]))
        if topic == imu_topic and imu_accel_scale != 1.0:
            scale_imu_acceleration(message, imu_accel_scale)
        if topic in (lidar_topic, imu_topic):
            serialized = serialize_message(message)
        writer.write(topic, serialized, int(receipt_time_ns))
        counts[topic] = counts.get(topic, 0) + 1

    if not counts.get(lidar_topic):
        raise ValueError(f"input bag contains no messages on {lidar_topic}")
    if not 0.0 <= scaled_max <= 1.0:
        raise ValueError(
            f"scaled per-point time maximum {scaled_max} s is outside [0, 1]; wrong scale?"
        )
    return {
        "schema": "unitree_pointcloud2_replay_normalization_v2",
        "input_bag": str(input_bag),
        "output_bag": str(output_bag),
        "lidar_topic": lidar_topic,
        "time_scale": scale,
        "imu_accel_scale": imu_accel_scale,
        "imu_accel_output_units": (
            "m/s^2" if np.isclose(imu_accel_scale, 9.80665) else "source units scaled"
        ),
        "header_aligned_to_bag_clock": align_header_to_bag_clock,
        "header_clock_mappings": header_mappings,
        "applied_header_clock_mapping_topic": (
            imu_topic if align_header_to_bag_clock else None
        ),
        "message_counts": counts,
        "point_count": point_count,
        "raw_time_min": raw_min,
        "raw_time_max": raw_max,
        "scaled_time_max_s": scaled_max,
        "preserved": [
            "topic names and types",
            "bag receipt timestamps",
            (
                "IMU payload fields except header stamp"
                if imu_accel_scale == 1.0
                else "IMU payload fields except header stamp and linear acceleration"
            ),
            "all PointCloud2 fields except time",
        ]
        + (
            ["LiDAR/IMU header cadence and cross-topic offset under one IMU affine map"]
            if align_header_to_bag_clock
            else ["LiDAR/IMU message headers"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bag", type=Path, required=True)
    parser.add_argument("--output-bag", type=Path, required=True)
    parser.add_argument(
        "--lidar-topic", default="/utlidar/cloud_livox_mid360"
    )
    parser.add_argument("--time-scale", type=float, default=1e-9)
    parser.add_argument("--imu-topic", default="/utlidar/imu_livox_mid360")
    parser.add_argument(
        "--imu-accel-scale",
        type=float,
        default=1.0,
        help="Multiply Livox linear acceleration by this factor (9.80665 converts g to m/s^2).",
    )
    parser.add_argument(
        "--align-header-to-bag-clock",
        action="store_true",
        help="Map LiDAR and IMU header clocks to the rosbag /clock epoch.",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = normalize_bag(
        args.input_bag,
        args.output_bag,
        lidar_topic=args.lidar_topic,
        scale=args.time_scale,
        align_header_to_bag_clock=args.align_header_to_bag_clock,
        imu_topic=args.imu_topic,
        imu_accel_scale=args.imu_accel_scale,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
