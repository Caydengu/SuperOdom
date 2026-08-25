"""Replay the production G1 KISS localization pipeline on a frozen archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from g1_root_state_bridge.clock_sync import fit_affine_lower_envelope_clock
from g1_root_state_bridge.dynamic_capture_io import iter_recorded_datagrams
from g1_root_state_bridge.kiss_registration import KissRegistration
from g1_root_state_bridge.live_pipeline import (
    ImuSampleGap,
    LiveLocalizationConfig,
    LiveLocalizationError,
    SelectedLocalizationPipeline,
    joint_sample_from_dynamic_packet,
)
from g1_root_state_bridge.protocol import REQUIRED_ROOT_FUSION_FLAGS, RootStateHealth


def _quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {name: float("nan") for name in ("mean", "p50", "p95", "p99", "maximum")}
    return {
        "mean": float(np.mean(finite)),
        "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "maximum": float(np.max(finite)),
    }


def _next_or_none(iterator: object) -> object | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def run_replay(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks_path = output_dir / "tracks.npz"
    treatments_path = output_dir / "treatments.jsonl"
    metrics_path = output_dir / "metrics.json"
    for path in (tracks_path, treatments_path, metrics_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    archive = np.load(args.scans, allow_pickle=False)
    points = np.asarray(archive["points_xyz_m"], dtype=np.float32)
    point_time = np.asarray(archive["point_relative_time_s"], dtype=np.float32)
    offsets = np.asarray(archive["scan_offsets"], dtype=np.int64)
    scan_source_time_ns = np.asarray(archive["source_time_ns"], dtype=np.int64)
    scan_receipt_time_ns = np.asarray(archive["receipt_time_ns"], dtype=np.int64)
    imu_source_time_ns = np.asarray(archive["imu_source_time_ns"], dtype=np.int64)
    imu_receipt_time_ns = np.asarray(archive["imu_receipt_time_ns"], dtype=np.int64)
    if scan_source_time_ns.shape != scan_receipt_time_ns.shape:
        raise ValueError("scan source and receipt timestamp arrays must have equal shape")
    if imu_source_time_ns.shape != imu_receipt_time_ns.shape:
        raise ValueError("IMU source and receipt timestamp arrays must have equal shape")
    # Match the live node exactly: the Livox IMU is the clock authority for both
    # IMU samples and PointCloud2 scan headers.  G1 source time can have a large
    # constant offset from host realtime, so using raw headers against mapped
    # LowState timestamps makes an otherwise valid capture look non-overlapping.
    imu_clock = fit_affine_lower_envelope_clock(
        imu_source_time_ns,
        imu_receipt_time_ns,
        lower_quantile=0.01,
    )
    scan_time_ns = np.asarray(
        [imu_clock.map_time_ns(value) for value in scan_source_time_ns],
        dtype=np.int64,
    )
    imu_time_ns = np.asarray(
        [imu_clock.map_time_ns(value) for value in imu_source_time_ns],
        dtype=np.int64,
    )
    imu_angular = np.asarray(archive["imu_angular_velocity_radps"], dtype=np.float64)
    if args.maximum_scans is not None:
        scan_time_ns = scan_time_ns[: args.maximum_scans]
        offsets = offsets[: scan_time_ns.size + 1]

    if args.synthetic_input_delivery_delay_ns < 0:
        raise ValueError("synthetic input delivery delay must be non-negative")
    lowstate_source_time_ns: list[int] = []
    lowstate_receipt_time_ns: list[int] = []
    for recorded in iter_recorded_datagrams(args.lowstate):
        lowstate_source_time_ns.append(int(recorded.packet.robot_stamp_ns))
        lowstate_receipt_time_ns.append(int(recorded.receipt_realtime_ns))
    if not lowstate_source_time_ns:
        raise ValueError("lowstate recording is empty")
    lowstate_clock = fit_affine_lower_envelope_clock(
        np.asarray(lowstate_source_time_ns, dtype=np.int64),
        np.asarray(lowstate_receipt_time_ns, dtype=np.int64),
        lower_quantile=0.01,
    )
    lowstate_first_ns = lowstate_clock.map_time_ns(lowstate_source_time_ns[0])
    lowstate_last_ns = lowstate_clock.map_time_ns(lowstate_source_time_ns[-1])

    pipeline = SelectedLocalizationPipeline(
        KissRegistration(
            voxel_size_m=args.voxel_size_m,
            minimum_range_m=args.minimum_range_m,
            maximum_range_m=args.maximum_range_m,
        ),
        LiveLocalizationConfig(
            gyro_bias_radps=tuple(args.gyro_bias_radps),
            maximum_imu_gap_ns=round(args.maximum_imu_gap_ms * 1e6),
            maximum_imu_bridge_gap_ns=round(args.maximum_imu_bridge_gap_ms * 1e6),
        ),
    )
    lowstate_iterator = iter(iter_recorded_datagrams(args.lowstate))
    pending_lowstate = _next_or_none(lowstate_iterator)
    if pending_lowstate is None:
        raise ValueError("lowstate recording is empty")
    pipeline.reset(int(pending_lowstate.packet.source_epoch))
    imu_index = 0
    positions: list[tuple[float, float, float]] = []
    quaternions_wxyz: list[tuple[float, float, float, float]] = []
    output_time_ns: list[int] = []
    healthy_output_time_ns: list[int] = []
    health_flags: list[int] = []
    pose_age_ms: list[float] = []
    runtime_rows: list[dict[str, float]] = []
    adaptive_threshold: list[float] = []
    deskew_excursion: list[float] = []
    dropped: dict[str, int] = {}
    imu_gap_bridges = 0
    imu_gap_restarts = 0
    wall_start_ns = time.perf_counter_ns()
    with treatments_path.open("x", encoding="utf-8") as treatments:
        treatments.write(
            json.dumps(
                {
                    "schema": "g1_amo_localization_treatment_track_v1",
                    "kind": "metadata",
                    "motive_online_input": False,
                    "alignment_reference_treatment": args.treatment_name,
                    "producer": "g1_kiss_live_localization_v1",
                    "scans": str(args.scans),
                    "lowstate": str(args.lowstate),
                },
                sort_keys=True,
            )
            + "\n"
        )
        for scan_index, scan_start_ns in enumerate(scan_time_ns):
            selected = slice(int(offsets[scan_index]), int(offsets[scan_index + 1]))
            relative = point_time[selected]
            scan_end_ns = int(scan_start_ns + round(float(np.max(relative)) * 1e9))
            while imu_index < imu_time_ns.size and int(imu_time_ns[imu_index]) <= scan_end_ns:
                try:
                    bridged = pipeline.append_imu(
                        int(imu_time_ns[imu_index]), imu_angular[imu_index]
                    )
                    imu_gap_bridges += int(bridged > 0)
                except ImuSampleGap:
                    pipeline.restart_imu_after_gap(
                        int(imu_time_ns[imu_index]), imu_angular[imu_index]
                    )
                    imu_gap_restarts += 1
                except LiveLocalizationError as error:
                    dropped[str(error)] = dropped.get(str(error), 0) + 1
                imu_index += 1
            if imu_index < imu_time_ns.size:
                try:
                    bridged = pipeline.append_imu(
                        int(imu_time_ns[imu_index]), imu_angular[imu_index]
                    )
                    imu_gap_bridges += int(bridged > 0)
                except ImuSampleGap:
                    pipeline.restart_imu_after_gap(
                        int(imu_time_ns[imu_index]), imu_angular[imu_index]
                    )
                    imu_gap_restarts += 1
                imu_index += 1
            while (
                pending_lowstate is not None
                and lowstate_clock.map_time_ns(int(pending_lowstate.packet.robot_stamp_ns))
                <= scan_end_ns
            ):
                mapped_joint_ns = lowstate_clock.map_time_ns(
                    int(pending_lowstate.packet.robot_stamp_ns)
                )
                pipeline.append_joint(
                    joint_sample_from_dynamic_packet(
                        pending_lowstate.packet,
                        receipt_time_ns=int(pending_lowstate.receipt_realtime_ns),
                        mapped_stamp_ns=mapped_joint_ns,
                    )
                )
                pending_lowstate = _next_or_none(lowstate_iterator)
            if pending_lowstate is not None:
                mapped_joint_ns = lowstate_clock.map_time_ns(
                    int(pending_lowstate.packet.robot_stamp_ns)
                )
                pipeline.append_joint(
                    joint_sample_from_dynamic_packet(
                        pending_lowstate.packet,
                        receipt_time_ns=int(pending_lowstate.receipt_realtime_ns),
                        mapped_stamp_ns=mapped_joint_ns,
                    )
                )
                pending_lowstate = _next_or_none(lowstate_iterator)
            try:
                output = pipeline.process_scan(
                    points[selected],
                    relative,
                    scan_start_time_ns=int(scan_start_ns),
                    publish_time_offset_ns=args.synthetic_input_delivery_delay_ns,
                )
            except LiveLocalizationError as error:
                dropped[str(error)] = dropped.get(str(error), 0) + 1
                if not output_time_ns and "IMU buffer does not bracket" in str(error):
                    pipeline.bootstrap_unpublished_scan(points[selected])
                continue
            packet = output.packet
            positions.append(packet.position)
            quaternions_wxyz.append(packet.quaternion_wxyz)
            output_time_ns.append(packet.estimate_time_ns)
            health_flags.append(int(packet.health_flags))
            packet_healthy = (
                packet.health_flags & REQUIRED_ROOT_FUSION_FLAGS
            ) == REQUIRED_ROOT_FUSION_FLAGS
            if packet_healthy:
                healthy_output_time_ns.append(packet.estimate_time_ns)
            pose_age_ms.append((packet.publish_time_ns - packet.estimate_time_ns) * 1e-6)
            runtime_rows.append(output.stage_runtime_ms)
            adaptive_threshold.append(output.adaptive_threshold)
            deskew_excursion.append(output.deskew_angular_excursion_deg)
            w, x, y, z = packet.quaternion_wxyz
            treatments.write(
                json.dumps(
                    {
                        "schema": "g1_amo_localization_treatment_track_v1",
                        "kind": "pose",
                        "treatment": args.treatment_name,
                        "source_time_ns": packet.estimate_time_ns,
                        "event_realtime_ns": packet.estimate_time_ns,
                        "source_lowstate_sequence": output.source_joint_sequence,
                        "source_lowstate_age_ns": abs(packet.joint_sync_gap_ns),
                        "frame_id": "kiss_icp_local",
                        "child_frame_id": "pelvis_navigation_yaw",
                        "position_xyz_m": list(packet.position),
                        "quaternion_xyzw": [x, y, z, w],
                        "orientation_fusion_healthy": bool(
                            packet.health_flags & RootStateHealth.HEADING_VALID
                        ),
                        "orientation_fusion_reason": (
                            "livox_navigation_heading"
                            if packet_healthy
                            else "bounded_livox_imu_gap_bridge"
                        ),
                        "yaw_innovation_rad": 0.0,
                        "adaptive_threshold": output.adaptive_threshold,
                        "runtime_ms": output.stage_runtime_ms["total"],
                        "deskew_valid": bool(
                            packet.health_flags
                            & RootStateHealth.INERTIAL_DESKEW_VALID
                        ),
                        "bridged_imu_gap_ns": output.bridged_imu_gap_ns,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    wall_elapsed_s = (time.perf_counter_ns() - wall_start_ns) * 1e-9
    stage_names = sorted({name for row in runtime_rows for name in row})
    stage_metrics = {
        name: _quantiles(np.asarray([row[name] for row in runtime_rows])) for name in stage_names
    }
    duration_s = float((scan_time_ns[-1] - scan_time_ns[0]) * 1e-9) if scan_time_ns.size > 1 else 0.0
    scan_end_time_ns = np.asarray(
        [
            int(scan_time_ns[index])
            + round(float(np.max(point_time[int(offsets[index]) : int(offsets[index + 1])])) * 1e9)
            for index in range(scan_time_ns.size)
        ],
        dtype=np.int64,
    )
    overlap = (scan_time_ns >= lowstate_first_ns) & (scan_end_time_ns <= lowstate_last_ns)
    overlap_scans = int(np.count_nonzero(overlap))
    overlap_outputs = int(
        np.count_nonzero(
            (np.asarray(output_time_ns, dtype=np.int64) >= lowstate_first_ns)
            & (np.asarray(output_time_ns, dtype=np.int64) <= lowstate_last_ns)
        )
    )
    healthy_times = np.asarray(healthy_output_time_ns, dtype=np.int64)
    healthy_overlap_outputs = int(
        np.count_nonzero(
            (healthy_times >= lowstate_first_ns) & (healthy_times <= lowstate_last_ns)
        )
    )
    maximum_healthy_output_gap_s = (
        float(np.max(np.diff(healthy_times))) * 1e-9
        if healthy_times.size >= 2
        else float("inf")
    )
    metrics: dict[str, object] = {
        "schema": "g1_kiss_live_localization_replay_v1",
        "scans_requested": int(scan_time_ns.size),
        "outputs": len(output_time_ns),
        "availability": len(output_time_ns) / int(scan_time_ns.size),
        "input_overlap_scans": overlap_scans,
        "input_overlap_outputs": overlap_outputs,
        "input_overlap_availability": overlap_outputs / overlap_scans if overlap_scans else 0.0,
        "healthy_outputs": len(healthy_output_time_ns),
        "healthy_input_overlap_outputs": healthy_overlap_outputs,
        "healthy_input_overlap_availability": (
            healthy_overlap_outputs / overlap_scans if overlap_scans else 0.0
        ),
        "maximum_healthy_output_gap_s": maximum_healthy_output_gap_s,
        "capture_boundary_scans_without_lowstate_coverage": int(scan_time_ns.size - overlap_scans),
        "dropped": dropped,
        "recorded_duration_s": duration_s,
        "wall_elapsed_s": wall_elapsed_s,
        "realtime_factor": duration_s / wall_elapsed_s if wall_elapsed_s > 0.0 else 0.0,
        "stage_runtime_ms": stage_metrics,
        "pose_age_ms": _quantiles(np.asarray(pose_age_ms, dtype=np.float64)),
        "synthetic_input_delivery_delay_ms": args.synthetic_input_delivery_delay_ns * 1e-6,
        "calibration_digest": pipeline.config.calibration_digest.hex(),
        "lowstate_clock": {
            "method": "affine_lower_envelope_q01",
            "slope": lowstate_clock.slope,
            "residual_p95_ns": lowstate_clock.residual_p95_ns,
            "lower_envelope_residual_ns": lowstate_clock.lower_envelope_residual_ns,
        },
        "imu_clock": {
            "method": "affine_lower_envelope_q01",
            "slope": imu_clock.slope,
            "residual_p95_ns": imu_clock.residual_p95_ns,
            "lower_envelope_residual_ns": imu_clock.lower_envelope_residual_ns,
        },
        "source_epoch": pipeline._source_epoch,
        "treatment": args.treatment_name,
        "maximum_imu_gap_ms": args.maximum_imu_gap_ms,
        "maximum_imu_bridge_gap_ms": args.maximum_imu_bridge_gap_ms,
        "imu_gap_bridges": imu_gap_bridges,
        "imu_gap_restarts": imu_gap_restarts,
    }
    np.savez_compressed(
        tracks_path,
        source_time_ns=np.asarray(output_time_ns, dtype=np.int64),
        position_xyz_m=np.asarray(positions, dtype=np.float64).reshape((-1, 3)),
        quaternion_wxyz=np.asarray(quaternions_wxyz, dtype=np.float64).reshape((-1, 4)),
        health_flags=np.asarray(health_flags, dtype=np.uint32),
        pose_age_ms=np.asarray(pose_age_ms, dtype=np.float64),
        adaptive_threshold=np.asarray(adaptive_threshold, dtype=np.float64),
        deskew_angular_excursion_deg=np.asarray(deskew_excursion, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(metrics, sort_keys=True)),
    )
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scans", type=Path, required=True)
    parser.add_argument("--lowstate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gyro-bias-radps", nargs=3, type=float, required=True)
    parser.add_argument("--treatment-name", default="kiss_live_selected_pelvis")
    parser.add_argument("--voxel-size-m", type=float, default=0.15)
    parser.add_argument("--minimum-range-m", type=float, default=0.5)
    parser.add_argument("--maximum-range-m", type=float, default=15.0)
    parser.add_argument("--maximum-imu-gap-ms", type=float, default=25.0)
    parser.add_argument("--maximum-imu-bridge-gap-ms", type=float, default=40.0)
    parser.add_argument(
        "--synthetic-input-delivery-delay-ns",
        type=int,
        default=5_000_000,
        help="added before measured processing time when constructing replay publish time",
    )
    parser.add_argument("--maximum-scans", type=int)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_replay(parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()
