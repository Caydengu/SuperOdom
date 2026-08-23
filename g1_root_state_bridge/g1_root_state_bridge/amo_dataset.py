"""Read-only synchronization and admission audit for G1 AMO/Motive captures."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from g1_root_state_bridge.dynamic_capture_io import iter_recorded_datagrams

MOTIVE_RAW_SCHEMA = "g1_optitrack_raw_v1"
DATASET_MANIFEST_SCHEMA = "g1_motive_localization_dataset_manifest_v1"
PELVIS_RIGID_BODY_ID = 39
PELVIS_RIGID_BODY_NAME = "G1-PELVIS-2296"
MAXIMUM_MARKER_ERROR_M = 0.005
LEG_JOINT_INDICES = tuple(range(12))


class AmoDatasetError(ValueError):
    """A captured stream violates the offline evaluation contract."""


@dataclass(frozen=True)
class AffineClockMapping:
    source_origin_ns: int
    target_origin_ns: int
    scale: float
    lower_envelope_residual_ns: float
    delay_p50_ns: float
    delay_p95_ns: float
    residual_p95_ns: float

    def map_ns(self, source_ns: np.ndarray) -> np.ndarray:
        centered = np.asarray(source_ns, dtype=np.float64) - self.source_origin_ns
        return (
            self.target_origin_ns
            + self.scale * centered
            + self.lower_envelope_residual_ns
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "source_origin_ns": self.source_origin_ns,
            "target_origin_ns": self.target_origin_ns,
            "scale": self.scale,
            "lower_envelope_residual_ns": self.lower_envelope_residual_ns,
            "delay_p50_ns": self.delay_p50_ns,
            "delay_p95_ns": self.delay_p95_ns,
            "residual_p95_ns": self.residual_p95_ns,
            "method": "affine_lower_envelope_q01",
        }


@dataclass(frozen=True)
class LowStateCapture:
    robot_stamp_ns: np.ndarray
    oslo_event_ns: np.ndarray
    receipt_realtime_ns: np.ndarray
    sequence: np.ndarray
    source_epoch: np.ndarray
    source_tick: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    imu_quaternion_wxyz: np.ndarray
    imu_gyroscope: np.ndarray
    imu_accelerometer: np.ndarray
    clock: AffineClockMapping


@dataclass(frozen=True)
class MotiveCapture:
    software_stamp_ns: np.ndarray
    oslo_event_ns: np.ndarray
    receipt_realtime_ns: np.ndarray
    frame_number: np.ndarray
    position_xyz_m: np.ndarray
    quaternion_xyzw: np.ndarray
    mean_marker_error_m: np.ndarray
    clock: AffineClockMapping
    total_frames: int
    rejected_tracking_frames: int
    rejected_marker_error_frames: int


@dataclass(frozen=True)
class WalkingBoundary:
    event_realtime_ns: int
    seconds_from_lowstate_start: float
    activity_threshold_rad_s: float
    evidence_window_s: float
    required_active_fraction: float
    first_confirming_window_end_s: float

    def as_dict(self) -> dict[str, object]:
        return {
            "event_realtime_ns": self.event_realtime_ns,
            "seconds_from_lowstate_start": self.seconds_from_lowstate_start,
            "activity_threshold_rad_s": self.activity_threshold_rad_s,
            "evidence_window_s": self.evidence_window_s,
            "required_active_fraction": self.required_active_fraction,
            "first_confirming_window_end_s": self.first_confirming_window_end_s,
            "detector": "leg_joint_velocity_rms_sustained_v1",
        }


def fit_affine_lower_envelope_clock(
    source_ns: np.ndarray,
    receipt_ns: np.ndarray,
    *,
    lower_quantile: float = 0.01,
) -> AffineClockMapping:
    """Fit source time into the receiver clock without assuming zero delay.

    Receipt time equals event time plus a non-negative transport/scheduling
    delay.  A centered affine fit estimates clock rate, and the lower residual
    quantile estimates the least-delayed event-time envelope.
    """

    source = np.asarray(source_ns, dtype=np.int64)
    receipt = np.asarray(receipt_ns, dtype=np.int64)
    if source.ndim != 1 or receipt.shape != source.shape or source.size < 3:
        raise AmoDatasetError("clock fit requires equal one-dimensional arrays")
    if not np.all(np.diff(source) > 0) or not np.all(np.diff(receipt) > 0):
        raise AmoDatasetError("clock samples must be strictly increasing")
    if not 0.0 <= lower_quantile < 0.5:
        raise AmoDatasetError("lower clock-envelope quantile must be in [0, 0.5)")

    source_origin = int(source[0])
    target_origin = int(receipt[0])
    x = source.astype(np.float64) - source_origin
    y = receipt.astype(np.float64) - target_origin
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 0.0:
        raise AmoDatasetError("clock source has no time extent")
    scale = float(np.dot(x_centered, y_centered) / denominator)
    if not 0.999 <= scale <= 1.001:
        raise AmoDatasetError(f"clock scale is implausible: {scale}")
    intercept = float(np.mean(y) - scale * np.mean(x))
    residual = y - (intercept + scale * x)
    lower = float(np.quantile(residual, lower_quantile))
    delay = residual - lower
    return AffineClockMapping(
        source_origin_ns=source_origin,
        target_origin_ns=target_origin,
        scale=scale,
        lower_envelope_residual_ns=intercept + lower,
        delay_p50_ns=float(np.quantile(delay, 0.50)),
        delay_p95_ns=float(np.quantile(delay, 0.95)),
        residual_p95_ns=float(
            np.quantile(np.abs(residual - np.median(residual)), 0.95)
        ),
    )


def load_lowstate(path: str | Path) -> LowStateCapture:
    records = list(iter_recorded_datagrams(path))
    if not records:
        raise AmoDatasetError(f"no lowstate records in {path}")
    robot_stamp_ns = np.asarray(
        [record.packet.robot_stamp_ns for record in records], dtype=np.int64
    )
    receipt_realtime_ns = np.asarray(
        [record.receipt_realtime_ns for record in records], dtype=np.int64
    )
    clock = fit_affine_lower_envelope_clock(robot_stamp_ns, receipt_realtime_ns)
    return LowStateCapture(
        robot_stamp_ns=robot_stamp_ns,
        oslo_event_ns=np.rint(clock.map_ns(robot_stamp_ns)).astype(np.int64),
        receipt_realtime_ns=receipt_realtime_ns,
        sequence=np.asarray(
            [record.packet.sequence for record in records], dtype=np.uint64
        ),
        source_epoch=np.asarray(
            [record.packet.source_epoch for record in records], dtype=np.uint64
        ),
        source_tick=np.asarray(
            [record.packet.source_tick for record in records], dtype=np.uint32
        ),
        joint_position=np.asarray(
            [record.packet.joint_position for record in records], dtype=np.float32
        ),
        joint_velocity=np.asarray(
            [record.packet.joint_velocity for record in records], dtype=np.float32
        ),
        imu_quaternion_wxyz=np.asarray(
            [record.packet.imu_quaternion_wxyz for record in records], dtype=np.float32
        ),
        imu_gyroscope=np.asarray(
            [record.packet.imu_gyroscope for record in records], dtype=np.float32
        ),
        imu_accelerometer=np.asarray(
            [record.packet.imu_accelerometer for record in records], dtype=np.float32
        ),
        clock=clock,
    )


def _iter_json_lines(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise AmoDatasetError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(payload, dict):
                raise AmoDatasetError(f"{path}:{line_number}: expected JSON object")
            yield payload


def load_motive(
    path: str | Path,
    *,
    expected_rigid_body_id: int = PELVIS_RIGID_BODY_ID,
    expected_rigid_body_name: str = PELVIS_RIGID_BODY_NAME,
) -> MotiveCapture:
    source = Path(path)
    frames: list[dict[str, object]] = []
    rejected_tracking = 0
    rejected_error = 0
    for payload in _iter_json_lines(source):
        if payload.get("record_type") == "metadata":
            if payload.get("schema") != MOTIVE_RAW_SCHEMA:
                raise AmoDatasetError(f"{source}: unsupported Motive metadata schema")
            if int(payload.get("requested_rigid_body_id", -1)) != expected_rigid_body_id:
                raise AmoDatasetError(f"{source}: wrong requested rigid-body ID")
            if payload.get("rigid_body_name") != expected_rigid_body_name:
                raise AmoDatasetError(f"{source}: wrong requested rigid-body name")
            continue
        if payload.get("record_type") != "frame":
            raise AmoDatasetError(f"{source}: unsupported Motive record type")
        if int(payload.get("rigid_body_id", -1)) != expected_rigid_body_id:
            raise AmoDatasetError(f"{source}: frame contains wrong rigid-body ID")
        if payload.get("rigid_body_name") != expected_rigid_body_name:
            raise AmoDatasetError(f"{source}: frame contains wrong rigid-body name")
        if not bool(payload.get("tracking_valid")):
            rejected_tracking += 1
            continue
        if float(payload["mean_marker_error_m"]) > MAXIMUM_MARKER_ERROR_M:
            rejected_error += 1
            continue
        frames.append(payload)
    if len(frames) < 3:
        raise AmoDatasetError(f"{source}: fewer than three admissible Motive frames")

    software_stamp_ns = np.rint(
        np.asarray(
            [frame["motive_software_time_s"] for frame in frames], dtype=np.float64
        )
        * 1e9
    ).astype(np.int64)
    receipt_realtime_ns = np.asarray(
        [frame["receipt_realtime_ns"] for frame in frames], dtype=np.int64
    )
    clock = fit_affine_lower_envelope_clock(software_stamp_ns, receipt_realtime_ns)
    return MotiveCapture(
        software_stamp_ns=software_stamp_ns,
        oslo_event_ns=np.rint(clock.map_ns(software_stamp_ns)).astype(np.int64),
        receipt_realtime_ns=receipt_realtime_ns,
        frame_number=np.asarray(
            [frame["frame_number"] for frame in frames], dtype=np.int64
        ),
        position_xyz_m=np.asarray(
            [frame["position_xyz_m_motive_native"] for frame in frames],
            dtype=np.float64,
        ),
        quaternion_xyzw=np.asarray(
            [frame["quaternion_xyzw_motive_native"] for frame in frames],
            dtype=np.float64,
        ),
        mean_marker_error_m=np.asarray(
            [frame["mean_marker_error_m"] for frame in frames], dtype=np.float64
        ),
        clock=clock,
        total_frames=len(frames) + rejected_tracking + rejected_error,
        rejected_tracking_frames=rejected_tracking,
        rejected_marker_error_frames=rejected_error,
    )


def detect_walking_boundary(
    capture: LowStateCapture,
    *,
    bin_width_s: float = 0.1,
    activity_threshold_rad_s: float = 0.18,
    evidence_window_s: float = 2.0,
    required_active_fraction: float = 0.8,
) -> WalkingBoundary:
    """Find the first sustained gait-like lower-body activity interval."""

    if bin_width_s <= 0.0 or evidence_window_s <= bin_width_s:
        raise AmoDatasetError("walking detector bin/window values are invalid")
    relative_s = (capture.oslo_event_ns - capture.oslo_event_ns[0]) * 1e-9
    bins = np.floor(relative_s / bin_width_s).astype(np.int64)
    bin_count = int(bins[-1]) + 1
    leg_rms = np.sqrt(
        np.mean(
            capture.joint_velocity[:, LEG_JOINT_INDICES].astype(np.float64) ** 2, axis=1
        )
    )
    activity_by_bin = np.zeros(bin_count, dtype=np.float64)
    for index in range(bin_count):
        values = leg_rms[bins == index]
        activity_by_bin[index] = np.median(values) if values.size else np.nan
    active = np.nan_to_num(activity_by_bin, nan=0.0) > activity_threshold_rad_s
    window_bins = max(2, round(evidence_window_s / bin_width_s))
    active_counts = np.convolve(
        active.astype(np.int64), np.ones(window_bins, dtype=np.int64), mode="valid"
    )
    required_count = int(np.ceil(required_active_fraction * window_bins))
    confirming = np.flatnonzero(active_counts >= required_count)
    if not confirming.size:
        raise AmoDatasetError("no sustained walking interval detected")
    window_start = int(confirming[0])
    within = np.flatnonzero(active[window_start : window_start + window_bins])
    if not within.size:
        raise AmoDatasetError("walking detector produced an empty active window")
    onset_bin = window_start + int(within[0])
    onset_s = onset_bin * bin_width_s
    return WalkingBoundary(
        event_realtime_ns=int(capture.oslo_event_ns[0] + round(onset_s * 1e9)),
        seconds_from_lowstate_start=float(onset_s),
        activity_threshold_rad_s=activity_threshold_rad_s,
        evidence_window_s=evidence_window_s,
        required_active_fraction=required_active_fraction,
        first_confirming_window_end_s=float((window_start + window_bins) * bin_width_s),
    )


def _load_bag_counts(run_dir: Path) -> dict[str, int]:
    metadata_path = run_dir / "lidar" / "data" / "live_input_probe" / "metadata.yaml"
    payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    information = payload["rosbag2_bagfile_information"]
    return {
        item["topic_metadata"]["name"]: int(item["message_count"])
        for item in information["topics_with_message_count"]
    }


def audit_run(run_dir: str | Path) -> tuple[dict[str, object], WalkingBoundary]:
    run = Path(run_dir).resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != DATASET_MANIFEST_SCHEMA:
        raise AmoDatasetError(f"{run}: unsupported dataset manifest")
    lowstate = load_lowstate(run / "lowstate" / "packets.bin")
    motive = load_motive(
        run / "motive" / "frames.jsonl",
        expected_rigid_body_id=int(manifest["rigid_body_id"]),
        expected_rigid_body_name=str(manifest["rigid_body_name"]),
    )
    boundary = detect_walking_boundary(lowstate)
    sequence_gaps = int(
        np.sum(np.maximum(np.diff(lowstate.sequence.astype(np.int64)) - 1, 0))
    )
    gap_indices = np.flatnonzero(np.diff(lowstate.sequence.astype(np.int64)) > 1)
    gap_intervals = [
        {
            "after_sequence": int(lowstate.sequence[index]),
            "missing_packets": int(
                lowstate.sequence[index + 1] - lowstate.sequence[index] - 1
            ),
            "event_gap_ms": float(
                (lowstate.oslo_event_ns[index + 1] - lowstate.oslo_event_ns[index])
                * 1e-6
            ),
            "seconds_from_lowstate_start": float(
                (lowstate.oslo_event_ns[index] - lowstate.oslo_event_ns[0]) * 1e-9
            ),
        }
        for index in gap_indices
    ]
    depth_path = run / "vision" / "capture" / "depth.npz"
    depth_chunks = sorted((run / "vision" / "capture").glob("depth_chunk_*.npz"))
    marker_quantiles = np.quantile(motive.mean_marker_error_m, [0.50, 0.95, 1.0])
    overlap_start = max(int(lowstate.oslo_event_ns[0]), int(motive.oslo_event_ns[0]))
    overlap_end = min(int(lowstate.oslo_event_ns[-1]), int(motive.oslo_event_ns[-1]))
    audit = {
        "schema": "g1_motive_amo_dataset_audit_v1",
        "run_dir": str(run),
        "raw_data_mutated": False,
        "motive_role": "evaluator_only",
        "absolute_pelvis_front_plane_claims_admitted": False,
        "absolute_claim_blocker": (
            "T_rigid_body_pelvis and Motive rigid-body axes are not independently validated"
        ),
        "relative_trajectory_claims_admitted": True,
        "lowstate": {
            "records": int(lowstate.sequence.size),
            "duration_s": float(
                (lowstate.oslo_event_ns[-1] - lowstate.oslo_event_ns[0]) * 1e-9
            ),
            "sequence_gaps": sequence_gaps,
            "gap_intervals": gap_intervals,
            "clock": lowstate.clock.as_dict(),
        },
        "motive": {
            "total_frames": motive.total_frames,
            "admitted_frames": int(motive.frame_number.size),
            "rejected_tracking_frames": motive.rejected_tracking_frames,
            "rejected_marker_error_frames": motive.rejected_marker_error_frames,
            "marker_error_m": {
                "p50": float(marker_quantiles[0]),
                "p95": float(marker_quantiles[1]),
                "maximum": float(marker_quantiles[2]),
            },
            "clock": motive.clock.as_dict(),
        },
        "livox_message_counts": _load_bag_counts(run),
        "rgbd": {
            "localization_admission": "diagnostic_only_pending_timing_and_extrinsics",
            "depth_chunk_count": len(depth_chunks),
            "monolithic_depth_present": depth_path.exists(),
            "monolithic_depth_zip_valid": depth_path.exists()
            and zipfile.is_zipfile(depth_path),
        },
        "common_event_time_overlap": {
            "start_realtime_ns": overlap_start,
            "end_realtime_ns": overlap_end,
            "duration_s": (overlap_end - overlap_start) * 1e-9,
        },
        "walking_boundary_candidate": boundary.as_dict(),
    }
    return audit, boundary
