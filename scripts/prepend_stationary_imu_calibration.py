#!/usr/bin/env python3
"""Prepend a same-robot stationary LiDAR/IMU interval to a normalized replay bag.

The G1 stress captures begin after AMO motion has already started, while both
SuperOdometry IMU initializers assume their first interval is stationary.  This
tool models the deployment contract "start localization, admit stationary IMU
calibration, then start locomotion" using a frozen stationary interval from the
G1-4123 calibration run.  It never changes the walk portion of the input bag.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scripts.normalize_unitree_pointcloud2_bag import normalize_time_field
except ModuleNotFoundError:  # Direct execution adds scripts/, not the repo root.
    from normalize_unitree_pointcloud2_bag import normalize_time_field


def shifted_time_ns(
    source_time_ns: int,
    *,
    source_origin_ns: int,
    target_origin_ns: int,
) -> int:
    return int(target_origin_ns) + (int(source_time_ns) - int(source_origin_ns))


def _stamp_ns(message: object) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


def _set_stamp_ns(message: object, stamp_ns: int) -> None:
    message.header.stamp.sec = int(stamp_ns) // 1_000_000_000
    message.header.stamp.nanosec = int(stamp_ns) % 1_000_000_000


def compose_bag(
    calibration_bag: Path,
    walk_bag: Path,
    output_bag: Path,
    *,
    imu_topic: str,
    lidar_topic: str,
    calibration_start_s: float,
    calibration_duration_s: float,
    calibration_accel_scale: float,
    transition_gap_ms: float,
) -> dict[str, object]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message, serialize_message
    from sensor_msgs.msg import Imu, PointCloud2

    if output_bag.exists():
        raise FileExistsError(f"refusing to overwrite output bag: {output_bag}")
    if calibration_start_s < 0.0 or calibration_duration_s < 1.1:
        raise ValueError("stationary calibration must start at/after zero and last at least 1.1 s")
    if not np.isfinite(calibration_accel_scale) or calibration_accel_scale <= 0.0:
        raise ValueError("calibration acceleration scale must be positive and finite")
    gap_ns = round(float(transition_gap_ms) * 1e6)
    if not 0 < gap_ns <= 100_000_000:
        raise ValueError("transition gap must be in (0, 100] ms")

    def open_reader(path: Path):
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
            rosbag2_py.ConverterOptions("cdr", "cdr"),
        )
        return reader

    walk_reader = open_reader(walk_bag)
    walk_topics = walk_reader.get_all_topics_and_types()
    walk_types = {topic.name: topic.type for topic in walk_topics}
    if walk_types.get(imu_topic) != "sensor_msgs/msg/Imu":
        raise ValueError(f"walk bag is missing Imu topic {imu_topic}")
    if walk_types.get(lidar_topic) != "sensor_msgs/msg/PointCloud2":
        raise ValueError(f"walk bag is missing PointCloud2 topic {lidar_topic}")
    first_walk_receipt_ns: int | None = None
    first_walk_imu_header_ns: int | None = None
    while walk_reader.has_next():
        topic, serialized, receipt_ns = walk_reader.read_next()
        if first_walk_receipt_ns is None:
            first_walk_receipt_ns = int(receipt_ns)
        if topic == imu_topic and first_walk_imu_header_ns is None:
            first_walk_imu_header_ns = _stamp_ns(deserialize_message(serialized, Imu))
        if first_walk_receipt_ns is not None and first_walk_imu_header_ns is not None:
            break
    if first_walk_receipt_ns is None or first_walk_imu_header_ns is None:
        raise ValueError("walk bag has no readable IMU samples")

    calibration_reader = open_reader(calibration_bag)
    calibration_types = {
        topic.name: topic.type for topic in calibration_reader.get_all_topics_and_types()
    }
    if calibration_types.get(imu_topic) != "sensor_msgs/msg/Imu":
        raise ValueError(f"calibration bag is missing Imu topic {imu_topic}")
    if calibration_types.get(lidar_topic) != "sensor_msgs/msg/PointCloud2":
        raise ValueError(f"calibration bag is missing PointCloud2 topic {lidar_topic}")
    all_start_ns: int | None = None
    selected: list[tuple[str, int, int, object]] = []
    requested_start_ns = round(calibration_start_s * 1e9)
    requested_end_ns = round((calibration_start_s + calibration_duration_s) * 1e9)
    while calibration_reader.has_next():
        topic, serialized, receipt_ns = calibration_reader.read_next()
        if topic not in (imu_topic, lidar_topic):
            continue
        message_type = Imu if topic == imu_topic else PointCloud2
        message = deserialize_message(serialized, message_type)
        source_ns = _stamp_ns(message)
        if all_start_ns is None and topic == imu_topic:
            all_start_ns = source_ns
        if all_start_ns is None:
            continue
        relative_ns = source_ns - all_start_ns
        if relative_ns < requested_start_ns:
            continue
        if relative_ns >= requested_end_ns:
            break
        selected.append((topic, source_ns, int(receipt_ns), message))
    imu_count = sum(topic == imu_topic for topic, *_ in selected)
    lidar_count = sum(topic == lidar_topic for topic, *_ in selected)
    if imu_count < 200 or lidar_count < 10:
        raise ValueError(
            "stationary calibration interval needs at least 200 IMU and 10 LiDAR samples"
        )

    selected_start_header_ns = min(source_ns for _, source_ns, _, _ in selected)
    selected_end_header_ns = max(source_ns for _, source_ns, _, _ in selected)
    selected_start_receipt_ns = min(receipt_ns for _, _, receipt_ns, _ in selected)
    selected_end_receipt_ns = max(receipt_ns for _, _, receipt_ns, _ in selected)
    selected_duration_ns = selected_end_header_ns - selected_start_header_ns
    selected_receipt_duration_ns = selected_end_receipt_ns - selected_start_receipt_ns
    prefix_header_origin_ns = (
        int(first_walk_imu_header_ns) - gap_ns - selected_duration_ns
    )
    prefix_receipt_origin_ns = (
        int(first_walk_receipt_ns) - gap_ns - selected_receipt_duration_ns
    )
    gyro_values: list[list[float]] = []
    accel_values: list[list[float]] = []
    output_bag.parent.mkdir(parents=True, exist_ok=True)
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output_bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    for topic in walk_topics:
        writer.create_topic(topic)

    prefix_counts = {imu_topic: 0, lidar_topic: 0}
    for topic, source_ns, receipt_ns, message in sorted(selected, key=lambda row: row[2]):
        output_header_ns = shifted_time_ns(
            source_ns,
            source_origin_ns=selected_start_header_ns,
            target_origin_ns=prefix_header_origin_ns,
        )
        output_receipt_ns = shifted_time_ns(
            receipt_ns,
            source_origin_ns=selected_start_receipt_ns,
            target_origin_ns=prefix_receipt_origin_ns,
        )
        _set_stamp_ns(message, output_header_ns)
        if topic == imu_topic:
            message.linear_acceleration.x *= calibration_accel_scale
            message.linear_acceleration.y *= calibration_accel_scale
            message.linear_acceleration.z *= calibration_accel_scale
            gyro_values.append(
                [
                    float(message.angular_velocity.x),
                    float(message.angular_velocity.y),
                    float(message.angular_velocity.z),
                ]
            )
            accel_values.append(
                [
                    float(message.linear_acceleration.x),
                    float(message.linear_acceleration.y),
                    float(message.linear_acceleration.z),
                ]
            )
        else:
            message.data, _ = normalize_time_field(
                message.data,
                fields=list(message.fields),
                width=message.width,
                height=message.height,
                point_step=message.point_step,
                row_step=message.row_step,
                is_bigendian=message.is_bigendian,
                scale=1e-9,
            )
        writer.write(topic, serialize_message(message), output_receipt_ns)
        prefix_counts[topic] += 1

    walk_reader = open_reader(walk_bag)
    walk_counts = {topic.name: 0 for topic in walk_topics}
    while walk_reader.has_next():
        topic, serialized, receipt_ns = walk_reader.read_next()
        writer.write(topic, serialized, int(receipt_ns))
        walk_counts[topic] = walk_counts.get(topic, 0) + 1

    gyro = np.asarray(gyro_values, dtype=np.float64)
    accel = np.asarray(accel_values, dtype=np.float64)
    return {
        "schema": "g1_stationary_lidar_imu_prefixed_replay_v1",
        "calibration_bag": str(calibration_bag),
        "walk_bag": str(walk_bag),
        "output_bag": str(output_bag),
        "imu_topic": imu_topic,
        "prefix_message_counts": prefix_counts,
        "prefix_duration_s": selected_duration_ns * 1e-9,
        "calibration_selection_s": [
            calibration_start_s,
            calibration_start_s + calibration_duration_s,
        ],
        "calibration_accel_scale": calibration_accel_scale,
        "transition_gap_ms": transition_gap_ms,
        "prefix_gyro_mean_radps": gyro.mean(axis=0).tolist(),
        "prefix_gyro_std_radps": gyro.std(axis=0).tolist(),
        "prefix_accel_mean_mps2": accel.mean(axis=0).tolist(),
        "prefix_accel_std_mps2": accel.std(axis=0).tolist(),
        "walk_message_counts": walk_counts,
        "walk_payload_contract": "byte-identical messages and receipt timestamps",
        "motive_online_input": False,
        "deployment_interpretation": (
            "localization is started while G1-4123 is stationary before AMO locomotion"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-bag", type=Path, required=True)
    parser.add_argument("--walk-bag", type=Path, required=True)
    parser.add_argument("--output-bag", type=Path, required=True)
    parser.add_argument("--imu-topic", default="/utlidar/imu_livox_mid360")
    parser.add_argument("--lidar-topic", default="/utlidar/cloud_livox_mid360")
    parser.add_argument("--calibration-start-s", type=float, default=0.0)
    parser.add_argument("--calibration-duration-s", type=float, default=2.0)
    parser.add_argument("--calibration-accel-scale", type=float, default=9.80665)
    parser.add_argument("--transition-gap-ms", type=float, default=5.0)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = compose_bag(
        args.calibration_bag,
        args.walk_bag,
        args.output_bag,
        imu_topic=args.imu_topic,
        lidar_topic=args.lidar_topic,
        calibration_start_s=args.calibration_start_s,
        calibration_duration_s=args.calibration_duration_s,
        calibration_accel_scale=args.calibration_accel_scale,
        transition_gap_ms=args.transition_gap_ms,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
