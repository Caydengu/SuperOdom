#!/usr/bin/env python3
"""Analyze live Livox/SuperOdometry bags without treating odometry as truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


def percentile(values: Sequence[float], quantile: float) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    position = (len(finite) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return finite[lower]
    weight = position - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "count": 0,
            "minimum": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "maximum": None,
            "mean": None,
        }
    return {
        "count": len(finite),
        "minimum": min(finite),
        "p50": percentile(finite, 0.50),
        "p95": percentile(finite, 0.95),
        "p99": percentile(finite, 0.99),
        "maximum": max(finite),
        "mean": sum(finite) / len(finite),
    }


def _rate_hz(timestamps_ns: Sequence[int]) -> float | None:
    if len(timestamps_ns) < 2:
        return None
    span_ns = timestamps_ns[-1] - timestamps_ns[0]
    if span_ns <= 0:
        return None
    return (len(timestamps_ns) - 1) * 1e9 / span_ns


def _nonmonotonic_count(timestamps_ns: Sequence[int]) -> int:
    return sum(current <= previous for previous, current in zip(timestamps_ns, timestamps_ns[1:]))


def _linear_slope(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y):
        raise ValueError("linear slope inputs differ in length")
    if len(x) < 2:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator == 0.0:
        return None
    return sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x, y)
    ) / denominator


def series_metrics(header_ns: Sequence[int], receipt_ns: Sequence[int]) -> dict[str, Any]:
    if len(header_ns) != len(receipt_ns):
        raise ValueError("header and receipt timestamp counts differ")
    interarrival_ms = [
        (current - previous) / 1e6
        for previous, current in zip(receipt_ns, receipt_ns[1:])
    ]
    header_interarrival_ms = [
        (current - previous) / 1e6
        for previous, current in zip(header_ns, header_ns[1:])
    ]
    receipt_age_ms = [
        (receipt - header) / 1e6
        for header, receipt in zip(header_ns, receipt_ns)
    ]
    elapsed_header_s = [
        (timestamp - header_ns[0]) / 1e9 for timestamp in header_ns
    ] if header_ns else []
    return {
        "count": len(header_ns),
        "header_nonmonotonic_count": _nonmonotonic_count(header_ns),
        "receipt_nonmonotonic_count": _nonmonotonic_count(receipt_ns),
        "header_rate_hz": _rate_hz(header_ns),
        "receipt_rate_hz": _rate_hz(receipt_ns),
        "header_span_s": ((header_ns[-1] - header_ns[0]) / 1e9) if len(header_ns) >= 2 else None,
        "receipt_span_s": ((receipt_ns[-1] - receipt_ns[0]) / 1e9) if len(receipt_ns) >= 2 else None,
        "receipt_age_ms": distribution(receipt_age_ms),
        "receipt_age_first_ms": receipt_age_ms[0] if receipt_age_ms else None,
        "receipt_age_last_ms": receipt_age_ms[-1] if receipt_age_ms else None,
        "receipt_age_growth_ms_per_s": _linear_slope(elapsed_header_s, receipt_age_ms),
        "header_interarrival_ms": distribution(header_interarrival_ms),
        "receipt_interarrival_ms": distribution(interarrival_ms),
    }


def _norm(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _yaw_from_quaternion(quaternion: Sequence[float]) -> float:
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _unwrap(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    output = [values[0]]
    for value in values[1:]:
        delta = value - output[-1]
        while delta > math.pi:
            value -= 2.0 * math.pi
            delta = value - output[-1]
        while delta < -math.pi:
            value += 2.0 * math.pi
            delta = value - output[-1]
        output.append(value)
    return output


def trajectory_metrics(
    positions: Sequence[Sequence[float]],
    quaternions: Sequence[Sequence[float]],
    timestamps_ns: Sequence[int] | None = None,
    warmup_s: float = 2.0,
) -> dict[str, Any]:
    if len(positions) != len(quaternions):
        raise ValueError("position and quaternion counts differ")
    if timestamps_ns is not None and len(timestamps_ns) != len(positions):
        raise ValueError("trajectory timestamp count differs")
    nonfinite = sum(
        not all(math.isfinite(float(value)) for value in (*position, *quaternion))
        for position, quaternion in zip(positions, quaternions)
    )
    trajectory_timestamps = list(timestamps_ns) if timestamps_ns is not None else [None] * len(positions)
    valid = [
        (tuple(map(float, position)), tuple(map(float, quaternion)), timestamp)
        for position, quaternion, timestamp in zip(positions, quaternions, trajectory_timestamps)
        if all(math.isfinite(float(value)) for value in (*position, *quaternion))
    ]
    if not valid:
        return {
            "count": len(positions),
            "nonfinite_sample_count": nonfinite,
            "terminal_translation_from_start_m": None,
            "maximum_translation_from_start_m": None,
            "maximum_consecutive_translation_m": None,
            "maximum_consecutive_translation_at_elapsed_s": None,
            "terminal_abs_yaw_from_start_deg": None,
            "maximum_abs_yaw_from_start_deg": None,
            "maximum_quaternion_norm_error": None,
            "post_warmup": None,
        }

    valid_positions = [item[0] for item in valid]
    valid_quaternions = [item[1] for item in valid]
    valid_timestamps = [item[2] for item in valid]
    origin = valid_positions[0]
    displacement = [
        _norm(tuple(value - initial for value, initial in zip(position, origin)))
        for position in valid_positions
    ]
    consecutive = [
        _norm(tuple(current_value - previous_value for current_value, previous_value in zip(current, previous)))
        for previous, current in zip(valid_positions, valid_positions[1:])
    ]
    yaw = _unwrap([_yaw_from_quaternion(quaternion) for quaternion in valid_quaternions])
    yaw_from_start_deg = [math.degrees(value - yaw[0]) for value in yaw]
    quaternion_norm_error = [abs(_norm(quaternion) - 1.0) for quaternion in valid_quaternions]
    maximum_translation_index = max(range(len(consecutive)), key=consecutive.__getitem__) + 1 if consecutive else 0
    translation_event_elapsed_s = None
    if valid_timestamps[0] is not None and valid_timestamps[maximum_translation_index] is not None:
        translation_event_elapsed_s = (
            int(valid_timestamps[maximum_translation_index]) - int(valid_timestamps[0])
        ) / 1e9

    post_warmup = None
    if valid_timestamps[0] is not None:
        warmup_indices = [
            index
            for index, timestamp in enumerate(valid_timestamps)
            if timestamp is not None
            and (int(timestamp) - int(valid_timestamps[0])) / 1e9 >= warmup_s
        ]
        if warmup_indices:
            start_index = warmup_indices[0]
            warmup_positions = valid_positions[start_index:]
            warmup_origin = warmup_positions[0]
            warmup_displacement = [
                _norm(tuple(value - initial for value, initial in zip(position, warmup_origin)))
                for position in warmup_positions
            ]
            warmup_consecutive = [
                _norm(tuple(current_value - previous_value for current_value, previous_value in zip(current, previous)))
                for previous, current in zip(warmup_positions, warmup_positions[1:])
            ]
            warmup_yaw = yaw[start_index:]
            warmup_yaw_from_start_deg = [
                math.degrees(value - warmup_yaw[0]) for value in warmup_yaw
            ]
            post_warmup = {
                "warmup_s": warmup_s,
                "sample_count": len(warmup_positions),
                "terminal_translation_from_warmup_start_m": warmup_displacement[-1],
                "maximum_translation_from_warmup_start_m": max(warmup_displacement),
                "maximum_consecutive_translation_m": max(warmup_consecutive, default=0.0),
                "terminal_abs_yaw_from_warmup_start_deg": abs(warmup_yaw_from_start_deg[-1]),
                "maximum_abs_yaw_from_warmup_start_deg": max(map(abs, warmup_yaw_from_start_deg)),
            }

    return {
        "count": len(positions),
        "nonfinite_sample_count": nonfinite,
        "terminal_translation_from_start_m": displacement[-1],
        "maximum_translation_from_start_m": max(displacement),
        "maximum_consecutive_translation_m": max(consecutive, default=0.0),
        "maximum_consecutive_translation_at_elapsed_s": translation_event_elapsed_s,
        "consecutive_translation_m": distribution(consecutive),
        "terminal_abs_yaw_from_start_deg": abs(yaw_from_start_deg[-1]),
        "maximum_abs_yaw_from_start_deg": max(map(abs, yaw_from_start_deg)),
        "consecutive_abs_yaw_deg": distribution(
            abs(current - previous)
            for previous, current in zip(yaw_from_start_deg, yaw_from_start_deg[1:])
        ),
        "maximum_quaternion_norm_error": max(quaternion_norm_error),
        "post_warmup": post_warmup,
    }


def optimization_stats_metrics(
    header_ns: Sequence[int],
    receipt_ns: Sequence[int],
    optimization_ms: Sequence[float],
    frame_processing_ms: Sequence[float],
) -> dict[str, Any]:
    if not (
        len(header_ns)
        == len(receipt_ns)
        == len(optimization_ms)
        == len(frame_processing_ms)
    ):
        raise ValueError("optimization statistics counts differ")
    result = series_metrics(header_ns, receipt_ns)
    result["optimization_ms"] = distribution(optimization_ms)
    result["frame_processing_ms"] = distribution(frame_processing_ms)
    return result


def correction_timing_metrics(
    *,
    reference_ns: Sequence[int],
    evidence_ns: Sequence[int],
    mapping_output_ns: Sequence[int],
    application_ns: Sequence[int],
    receipt_ns: Sequence[int],
    sequences: Sequence[int],
    reset_ids: Sequence[int],
    valid: Sequence[bool],
) -> dict[str, Any]:
    lengths = {
        len(reference_ns),
        len(evidence_ns),
        len(mapping_output_ns),
        len(application_ns),
        len(receipt_ns),
        len(sequences),
        len(reset_ids),
        len(valid),
    }
    if len(lengths) != 1:
        raise ValueError("correction timing counts differ")
    timing_order_violations = sum(
        not reference <= evidence <= output <= application <= receipt
        for reference, evidence, output, application, receipt in zip(
            reference_ns,
            evidence_ns,
            mapping_output_ns,
            application_ns,
            receipt_ns,
        )
    )
    return {
        "count": len(reference_ns),
        "valid_count": sum(bool(value) for value in valid),
        "invalid_count": sum(not bool(value) for value in valid),
        "timing_order_violation_count": timing_order_violations,
        "reference_nonmonotonic_count": _nonmonotonic_count(reference_ns),
        "evidence_nonmonotonic_count": _nonmonotonic_count(evidence_ns),
        "sequence_nonmonotonic_count": _nonmonotonic_count(sequences),
        "reset_ids": sorted(set(int(value) for value in reset_ids)),
        "scan_duration_ms": distribution(
            (evidence - reference) / 1e6
            for reference, evidence in zip(reference_ns, evidence_ns)
        ),
        "correction_evidence_gap_ms": distribution(
            (current - previous) / 1e6
            for previous, current in zip(evidence_ns, evidence_ns[1:])
        ),
        "mapping_delay_from_newest_observation_ms": distribution(
            (output - evidence) / 1e6
            for evidence, output in zip(evidence_ns, mapping_output_ns)
        ),
        "estimator_application_delay_ms": distribution(
            (application - output) / 1e6
            for output, application in zip(mapping_output_ns, application_ns)
        ),
        "bridge_transport_delay_ms": distribution(
            (receipt - application) / 1e6
            for application, receipt in zip(application_ns, receipt_ns)
        ),
        "evidence_age_at_bridge_receipt_ms": distribution(
            (receipt - evidence) / 1e6
            for evidence, receipt in zip(evidence_ns, receipt_ns)
        ),
    }


def timestamp_correspondence(source_ns: Sequence[int], output_ns: Sequence[int]) -> dict[str, Any]:
    source = set(source_ns)
    output = set(output_ns)
    matched = len(source & output)
    return {
        "source_unique_count": len(source),
        "output_unique_count": len(output),
        "output_matched_count": matched,
        "output_unmatched_count": len(output) - matched,
        "output_match_fraction": matched / len(output) if output else None,
    }


def _stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def analyze_bag(bag_path: Path, launch_wall_ns: int | None = None) -> dict[str, Any]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    message_classes = {topic: get_message(type_name) for topic, type_name in topic_types.items()}

    imu_header: list[int] = []
    imu_receipt: list[int] = []
    imu_measurements: list[float] = []
    lidar_header: list[int] = []
    lidar_receipt: list[int] = []
    lidar_timebase: list[int] = []
    lidar_newest_time: list[int] = []
    lidar_point_count: list[int] = []
    lidar_point_num_mismatch = 0
    lidar_header_timebase_delta: list[int] = []
    lidar_offset_min: list[int] = []
    lidar_offset_max: list[int] = []
    odom_header: list[int] = []
    odom_receipt: list[int] = []
    positions: list[tuple[float, float, float]] = []
    quaternions: list[tuple[float, float, float, float]] = []
    laser_odom_header: list[int] = []
    laser_odom_receipt: list[int] = []
    laser_positions: list[tuple[float, float, float]] = []
    laser_quaternions: list[tuple[float, float, float, float]] = []
    lidar_correction_reference: list[int] = []
    lidar_correction_evidence: list[int] = []
    lidar_correction_output: list[int] = []
    lidar_correction_receipt: list[int] = []
    lidar_correction_sequence: list[int] = []
    state_correction_reference: list[int] = []
    state_correction_evidence: list[int] = []
    state_correction_mapping_output: list[int] = []
    state_correction_application: list[int] = []
    state_correction_receipt: list[int] = []
    state_correction_sequence: list[int] = []
    state_correction_reset_id: list[int] = []
    state_correction_valid: list[bool] = []
    stats_header: list[int] = []
    stats_receipt: list[int] = []
    stats_optimization_ms: list[float] = []
    stats_processing_ms: list[float] = []
    stats_iterations: list[int] = []
    stats_translation_from_last: list[float] = []
    stats_rotation_from_last: list[float] = []
    health_values: list[bool] = []
    prediction_sources: list[str] = []

    while reader.has_next():
        topic, serialized, receipt_ns = reader.read_next()
        if topic not in message_classes:
            continue
        message = deserialize_message(serialized, message_classes[topic])
        if topic == "/livox/imu":
            imu_header.append(_stamp_ns(message.header.stamp))
            imu_receipt.append(int(receipt_ns))
            imu_measurements.extend(
                (
                    message.angular_velocity.x,
                    message.angular_velocity.y,
                    message.angular_velocity.z,
                    message.linear_acceleration.x,
                    message.linear_acceleration.y,
                    message.linear_acceleration.z,
                )
            )
        elif topic == "/livox/lidar":
            header_ns = _stamp_ns(message.header.stamp)
            timebase = int(message.timebase)
            offset_min: int | None = None
            offset_max: int | None = None
            for point in message.points:
                offset = int(point.offset_time)
                offset_min = offset if offset_min is None else min(offset_min, offset)
                offset_max = offset if offset_max is None else max(offset_max, offset)
            offset_min = 0 if offset_min is None else offset_min
            offset_max = 0 if offset_max is None else offset_max
            lidar_header.append(header_ns)
            lidar_receipt.append(int(receipt_ns))
            lidar_timebase.append(timebase)
            lidar_newest_time.append(timebase + offset_max)
            lidar_point_count.append(len(message.points))
            lidar_point_num_mismatch += int(int(message.point_num) != len(message.points))
            lidar_header_timebase_delta.append(header_ns - timebase)
            lidar_offset_min.append(offset_min)
            lidar_offset_max.append(offset_max)
        elif topic == "/state_estimation":
            odom_header.append(_stamp_ns(message.header.stamp))
            odom_receipt.append(int(receipt_ns))
            position = message.pose.pose.position
            orientation = message.pose.pose.orientation
            positions.append((position.x, position.y, position.z))
            quaternions.append((orientation.x, orientation.y, orientation.z, orientation.w))
        elif topic == "/laser_odometry":
            laser_odom_header.append(_stamp_ns(message.header.stamp))
            laser_odom_receipt.append(int(receipt_ns))
            position = message.pose.pose.position
            orientation = message.pose.pose.orientation
            laser_positions.append((position.x, position.y, position.z))
            laser_quaternions.append((orientation.x, orientation.y, orientation.z, orientation.w))
        elif topic == "/lidar_correction":
            lidar_correction_reference.append(_stamp_ns(message.odometry.header.stamp))
            lidar_correction_evidence.append(
                _stamp_ns(message.newest_observation_stamp)
            )
            lidar_correction_output.append(_stamp_ns(message.output_stamp))
            lidar_correction_receipt.append(int(receipt_ns))
            lidar_correction_sequence.append(int(message.sequence))
        elif topic == "/state_estimation_correction":
            state_correction_reference.append(_stamp_ns(message.header.stamp))
            state_correction_evidence.append(
                _stamp_ns(message.newest_observation_stamp)
            )
            state_correction_mapping_output.append(
                _stamp_ns(message.mapping_output_stamp)
            )
            state_correction_application.append(_stamp_ns(message.application_stamp))
            state_correction_receipt.append(int(receipt_ns))
            state_correction_sequence.append(int(message.sequence))
            state_correction_reset_id.append(int(message.reset_id))
            state_correction_valid.append(bool(message.valid))
        elif topic == "/super_odometry_stats":
            stats_header.append(_stamp_ns(message.header.stamp))
            stats_receipt.append(int(receipt_ns))
            stats_optimization_ms.append(float(message.time_elapsed))
            stats_processing_ms.append(float(message.latency))
            stats_iterations.append(int(message.n_iterations))
            stats_translation_from_last.append(float(message.translation_from_last))
            stats_rotation_from_last.append(float(message.rotation_from_last))
        elif topic == "/state_estimation_health":
            health_values.append(bool(message.data))
        elif topic == "/prediction_source":
            prediction_sources.append(str(message.data))

    result: dict[str, Any] = {
        "bag_path": str(bag_path),
        "topic_types": topic_types,
        "imu": series_metrics(imu_header, imu_receipt) if imu_header else None,
        "lidar": series_metrics(lidar_header, lidar_receipt) if lidar_header else None,
        "state_estimation": series_metrics(odom_header, odom_receipt) if odom_header else None,
        "laser_odometry": series_metrics(laser_odom_header, laser_odom_receipt) if laser_odom_header else None,
        "lidar_correction": {
            **series_metrics(lidar_correction_reference, lidar_correction_receipt),
            "sequence_nonmonotonic_count": _nonmonotonic_count(
                lidar_correction_sequence
            ),
            "scan_duration_ms": distribution(
                (evidence - reference) / 1e6
                for reference, evidence in zip(
                    lidar_correction_reference, lidar_correction_evidence
                )
            ),
            "mapping_delay_from_newest_observation_ms": distribution(
                (output - evidence) / 1e6
                for evidence, output in zip(
                    lidar_correction_evidence, lidar_correction_output
                )
            ),
            "output_transport_delay_ms": distribution(
                (receipt - output) / 1e6
                for output, receipt in zip(
                    lidar_correction_output, lidar_correction_receipt
                )
            ),
        } if lidar_correction_reference else None,
        "state_estimation_correction": correction_timing_metrics(
            reference_ns=state_correction_reference,
            evidence_ns=state_correction_evidence,
            mapping_output_ns=state_correction_mapping_output,
            application_ns=state_correction_application,
            receipt_ns=state_correction_receipt,
            sequences=state_correction_sequence,
            reset_ids=state_correction_reset_id,
            valid=state_correction_valid,
        ) if state_correction_reference else None,
        "super_odometry_stats": optimization_stats_metrics(
            stats_header,
            stats_receipt,
            stats_optimization_ms,
            stats_processing_ms,
        ) if stats_header else None,
        "state_estimation_health": {
            "count": len(health_values),
            "true_count": sum(health_values),
            "false_count": len(health_values) - sum(health_values),
        } if health_values else None,
        "prediction_source_counts": {
            value: prediction_sources.count(value) for value in sorted(set(prediction_sources))
        } if prediction_sources else None,
    }
    if imu_header:
        result["imu"]["nonfinite_measurement_count"] = sum(
            not math.isfinite(float(value)) for value in imu_measurements
        )
    if lidar_header:
        result["lidar"].update(
            {
                "timebase_nonmonotonic_count": _nonmonotonic_count(lidar_timebase),
                "header_minus_timebase_ns": distribution(lidar_header_timebase_delta),
                "point_count": distribution(lidar_point_count),
                "point_num_mismatch_count": lidar_point_num_mismatch,
                "minimum_point_offset_ns": distribution(lidar_offset_min),
                "maximum_point_offset_ns": distribution(lidar_offset_max),
                "newest_point_receipt_age_ms": distribution(
                    (receipt - newest) / 1e6
                    for newest, receipt in zip(lidar_newest_time, lidar_receipt)
                ),
            }
        )
    if odom_header:
        result["state_estimation"]["trajectory"] = trajectory_metrics(
            positions, quaternions, timestamps_ns=odom_header
        )
        result["state_estimation"]["imu_timestamp_correspondence"] = timestamp_correspondence(
            imu_header, odom_header
        )
        if launch_wall_ns is not None:
            result["state_estimation"]["first_receipt_after_launch_ms"] = (
                odom_receipt[0] - launch_wall_ns
            ) / 1e6
    if laser_odom_header:
        result["laser_odometry"]["trajectory"] = trajectory_metrics(
            laser_positions, laser_quaternions, timestamps_ns=laser_odom_header
        )
    if stats_header:
        result["super_odometry_stats"].update(
            {
                "n_iterations": distribution(stats_iterations),
                "translation_from_last_m": distribution(stats_translation_from_last),
                "rotation_from_last_rad": distribution(stats_rotation_from_last),
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--launch-wall-ns-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    launch_wall_ns = None
    if args.launch_wall_ns_file:
        launch_wall_ns = int(args.launch_wall_ns_file.read_text(encoding="utf-8").strip())
    result = analyze_bag(args.bag, launch_wall_ns=launch_wall_ns)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
