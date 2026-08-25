#!/usr/bin/env python3
"""Replay registered KISS-local evidence through the production map engine."""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from g1_root_state_bridge.map_protocol import serialize_map_correction_v1
from g1_root_state_bridge.structural_map_localization import (
    AutomaticMapCorrectionEngine,
    RegisteredMapEvidenceBuffer,
    StructuralMapConfig,
    StructuralMapLocalizationError,
    TimedLocalPose,
    TimedRegisteredCloud,
)
from g1_root_state_bridge.ui_initialization import load_ui_map_initialization


TREATMENTS = (
    "initialization-only",
    "stopped-only-full-se2",
    "periodic-full-se2",
)


def _load_odometry(
    path: Path, topic: str, treatment: str | None
) -> list[dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            is_odometry = row.get("kind") == "odometry" and row.get("topic") == topic
            is_treatment = row.get("kind") == "pose" and row.get("treatment") == treatment
            if is_odometry:
                if row.get("frame_id") != "kiss_local":
                    raise ValueError("odometry frame must be kiss_local")
                rows.append(row)
            elif is_treatment:
                rows.append(
                    {
                        **row,
                        "source_time_ns": int(
                            row.get("source_time_ns", row["event_realtime_ns"])
                        ),
                        "frame_id": "kiss_local",
                    }
                )
    if len(rows) < 3:
        raise ValueError(f"missing odometry topic {topic!r}")
    time_ns = np.asarray([row["source_time_ns"] for row in rows], dtype=np.int64)
    if not np.all(np.diff(time_ns) > 0):
        raise ValueError("odometry source time did not strictly increase")
    return rows


def _json_safe(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _window_is_stationary(
    odometry: list[dict[str, object]],
    source_time_ns: np.ndarray,
    *,
    end_time_ns: int,
    window_sec: float,
    maximum_linear_speed_mps: float,
    maximum_yaw_rate_radps: float,
) -> tuple[bool, dict[str, float]]:
    start_time_ns = end_time_ns - round(window_sec * 1e9)
    first = int(np.searchsorted(source_time_ns, start_time_ns, side="left"))
    last = int(np.searchsorted(source_time_ns, end_time_ns, side="right"))
    selected = odometry[first:last]
    coverage_sec = (
        0.0
        if len(selected) < 2
        else (int(selected[-1]["source_time_ns"]) - int(selected[0]["source_time_ns"]))
        * 1e-9
    )
    if len(selected) < 3 or coverage_sec < 0.95 * window_sec:
        return False, {"coverage_sec": coverage_sec}
    linear = np.asarray(
        [row["linear_velocity_xyz_mps"][:2] for row in selected], dtype=np.float64
    )
    yaw_rate = np.asarray(
        [row["angular_velocity_xyz_radps"][2] for row in selected], dtype=np.float64
    )
    speed_p90 = float(np.quantile(np.linalg.norm(linear, axis=1), 0.90))
    yaw_rate_p90 = float(np.quantile(np.abs(yaw_rate), 0.90))
    audit = {
        "coverage_sec": coverage_sec,
        "linear_speed_p90_mps": speed_p90,
        "yaw_rate_p90_radps": yaw_rate_p90,
    }
    return (
        speed_p90 <= maximum_linear_speed_mps
        and yaw_rate_p90 <= maximum_yaw_rate_radps,
        audit,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--map-key", default="map_xy_all_5cm")
    parser.add_argument("--map-version", type=int, default=1)
    parser.add_argument("--map-epoch", type=int, default=1)
    parser.add_argument("--registered-clouds", type=Path, required=True)
    parser.add_argument("--odometry", type=Path, required=True)
    parser.add_argument("--odometry-topic", default="/g1/localization/pelvis_odom")
    parser.add_argument("--odometry-treatment")
    parser.add_argument("--local-source-epoch", type=int, required=True)
    parser.add_argument("--treatment", choices=TREATMENTS, default="initialization-only")
    parser.add_argument("--correction-period-sec", type=float, default=5.0)
    parser.add_argument("--stationary-window-sec", type=float, default=5.0)
    parser.add_argument("--stationary-linear-speed-mps", type=float, default=0.05)
    parser.add_argument("--stationary-yaw-rate-radps", type=float, default=0.15)
    parser.add_argument(
        "--initialization-receipt",
        type=Path,
        help="accepted Gio UI/headless bounded-Snap receipt; skips global search",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packets-output", type=Path, required=True)
    args = parser.parse_args()
    if args.correction_period_sec <= 0.0 or args.stationary_window_sec <= 0.0:
        parser.error("correction period and stationary window must be positive")
    if args.output.exists() or args.packets_output.exists():
        raise FileExistsError("refusing to overwrite map-replay output")

    config = StructuralMapConfig()
    engine = AutomaticMapCorrectionEngine(
        map_path=args.map,
        map_key=args.map_key,
        map_version=args.map_version,
        map_epoch=args.map_epoch,
        config=config,
    )
    engine.bind_local_epoch(args.local_source_epoch)
    initialization = None
    if args.initialization_receipt is not None:
        initialization = load_ui_map_initialization(
            args.initialization_receipt,
            expected_structural_map_sha256=engine.map_digest.hex(),
            minimum_fitness=config.minimum_inlier_fraction,
            maximum_rmse_m=config.maximum_rmse_m,
            minimum_eigenvalue=config.minimum_observability_eigenvalue,
            maximum_condition_number=config.maximum_observability_condition_number,
        )
    buffer = RegisteredMapEvidenceBuffer(maximum_history_sec=15.0)
    odometry = _load_odometry(
        args.odometry, args.odometry_topic, args.odometry_treatment
    )
    odom_index = 0
    odometry_time_ns = np.asarray(
        [row["source_time_ns"] for row in odometry], dtype=np.int64
    )
    attempts: list[dict[str, object]] = []
    packets: list[dict[str, object]] = []
    last_attempt_ns = 0
    stopped_episode_applied = False
    last_motion_evidence_ns: int | None = None
    with np.load(args.registered_clouds, allow_pickle=False) as archive:
        points = np.asarray(archive["points_xyz_m"], dtype=np.float32)
        offsets = np.asarray(archive["cloud_offsets"], dtype=np.int64)
        cloud_time = np.asarray(archive["source_time_ns"], dtype=np.int64)
        frame = np.asarray(archive["frame_id"])
    for index, evidence_ns in enumerate(cloud_time):
        while odom_index < len(odometry) and int(
            odometry[odom_index]["source_time_ns"]
        ) <= int(evidence_ns):
            row = odometry[odom_index]
            buffer.append_pose(
                TimedLocalPose(
                    int(row["source_time_ns"]),
                    str(row["frame_id"]),
                    np.asarray(row["position_xyz_m"][:2], dtype=np.float64),
                )
            )
            odom_index += 1
        buffer.append_cloud(
            TimedRegisteredCloud(
                int(evidence_ns),
                str(frame[index]),
                points[offsets[index] : offsets[index + 1]],
            )
        )
        if (
            engine.rotation is None
            and initialization is not None
            and int(evidence_ns) < initialization.evidence_time_ns
        ):
            continue
        if engine.rotation is None:
            window_sec = config.global_window_sec
        else:
            if args.treatment == "initialization-only":
                continue
            window_sec = args.correction_period_sec
            if args.treatment == "periodic-full-se2":
                if int(evidence_ns) - last_attempt_ns < round(
                    args.correction_period_sec * 1e9
                ):
                    continue
            else:
                if last_motion_evidence_ns is None:
                    last_motion_evidence_ns = int(cloud_time[0])
                _, recent_motion_audit = _window_is_stationary(
                    odometry,
                    odometry_time_ns,
                    end_time_ns=int(evidence_ns),
                    window_sec=1.0,
                    maximum_linear_speed_mps=args.stationary_linear_speed_mps,
                    maximum_yaw_rate_radps=args.stationary_yaw_rate_radps,
                )
                # Reset the episode only on positive motion evidence.  A 10 Hz
                # odometry stream can cover slightly under one nominal second;
                # missing coverage must not manufacture a new stop episode.
                moving = (
                    recent_motion_audit.get("coverage_sec", 0.0) >= 0.75
                    and (
                        recent_motion_audit.get("linear_speed_p90_mps", 0.0)
                        > 2.0 * args.stationary_linear_speed_mps
                        or recent_motion_audit.get("yaw_rate_p90_radps", 0.0)
                        > 2.0 * args.stationary_yaw_rate_radps
                    )
                )
                if moving:
                    last_motion_evidence_ns = int(evidence_ns)
                    stopped_episode_applied = False
                    continue
                if int(evidence_ns) - last_motion_evidence_ns < round(
                    args.stationary_window_sec * 1e9
                ):
                    continue
                if stopped_episode_applied:
                    continue
                fresh_stationary, stationarity_audit = _window_is_stationary(
                    odometry,
                    odometry_time_ns,
                    end_time_ns=int(evidence_ns),
                    window_sec=args.stationary_window_sec,
                    maximum_linear_speed_mps=args.stationary_linear_speed_mps,
                    maximum_yaw_rate_radps=args.stationary_yaw_rate_radps,
                )
                if not fresh_stationary:
                    continue
                window_sec = args.stationary_window_sec
        end_to_end_start = time.perf_counter_ns()
        previous_rotation = None if engine.rotation is None else engine.rotation.copy()
        previous_translation = (
            None if engine.translation_m is None else engine.translation_m.copy()
        )
        previous_sequence = engine.sequence
        try:
            if engine.rotation is None and initialization is not None:
                audit = {
                    "schema": "g1_registered_map_query_audit_v1",
                    "source": "accepted_ui_initialization_receipt",
                    "start_source_time_ns": initialization.reference_time_ns,
                    "end_source_time_ns": initialization.evidence_time_ns,
                }
                attempt = engine.initialize_from_ui(
                    map_T_local=initialization.map_T_local,
                    fitness=initialization.fitness,
                    rmse_m=initialization.rmse_m,
                    min_eig=initialization.min_eig,
                    cond_number=initialization.cond_number,
                    reference_time_ns=initialization.reference_time_ns,
                    evidence_time_ns=initialization.evidence_time_ns,
                    # Offline replay preserves the evidence ordering without
                    # pretending wall-clock latency from another session.
                    application_time_ns=initialization.evidence_time_ns + 1,
                )
            else:
                query, audit = buffer.structural_query(
                    end_source_time_ns=int(evidence_ns),
                    window_sec=window_sec,
                    config=config,
                )
                if engine.rotation is None:
                    attempt = engine.initialize(
                        query,
                        reference_time_ns=int(audit["start_source_time_ns"]),
                        evidence_time_ns=int(evidence_ns),
                        # The packet is replaced below with the measured replay age.
                        application_time_ns=int(evidence_ns) + 1,
                    )
                else:
                    pose = buffer.pose_at(
                        int(evidence_ns),
                        maximum_age_ns=config.maximum_root_evidence_gap_ns,
                    )
                    attempt = engine.track(
                        query,
                        local_position_xy_m=pose.position_xy_m,
                        reference_time_ns=int(audit["start_source_time_ns"]),
                        evidence_time_ns=int(evidence_ns),
                        application_time_ns=int(evidence_ns) + 1,
                    )
        except StructuralMapLocalizationError:
            continue
        end_to_end_ms = (time.perf_counter_ns() - end_to_end_start) * 1e-6
        last_attempt_ns = int(evidence_ns)
        packet = attempt.packet
        rejection_reason = attempt.rejection_reason
        accepted = attempt.accepted
        if accepted and end_to_end_ms > config.maximum_input_age_ns * 1e-6:
            accepted = False
            packet = None
            rejection_reason = "map_evidence_stale_before_publication"
            engine.rotation = previous_rotation
            engine.translation_m = previous_translation
            engine.sequence = previous_sequence
        if packet is not None:
            if attempt.kind == "heading_only_tracking":
                candidate_transform = np.asarray(
                    attempt.report["candidate_map_T_local"], dtype=np.float64
                )
                packet = dataclasses.replace(packet, map_T_local=candidate_transform)
                engine.rotation = candidate_transform[:2, :2].T.copy()
                engine.translation_m = candidate_transform[:2, 3].copy()
            application_ns = int(evidence_ns) + max(1, round(end_to_end_ms * 1e6))
            packet = dataclasses.replace(
                packet,
                application_time_ns=application_ns,
                publish_time_ns=application_ns,
            )
            payload = serialize_map_correction_v1(packet)
            packets.append(
                {
                    "schema": "g1_map_correction_replay_packet_v1",
                    "kind": attempt.kind,
                    "treatment": args.treatment,
                    "evidence_time_ns": int(evidence_ns),
                    "end_to_end_runtime_ms": end_to_end_ms,
                    "payload_hex": payload.hex(),
                }
            )
        icp = attempt.report.get("icp", attempt.report)
        attempts.append(
            {
                "kind": attempt.kind,
                "treatment": args.treatment,
                "accepted": accepted,
                "rejection_reason": rejection_reason,
                "reference_time_ns": attempt.reference_time_ns,
                "evidence_time_ns": int(evidence_ns),
                "algorithm_runtime_ms": attempt.runtime_ms,
                "end_to_end_runtime_ms": end_to_end_ms,
                "query_audit": audit,
                "stationarity_audit": (
                    stationarity_audit
                    if args.treatment == "stopped-only-full-se2"
                    and attempt.kind == "heading_only_tracking"
                    else None
                ),
                "report": _json_safe(icp),
            }
        )
        if (
            args.treatment == "stopped-only-full-se2"
            and attempt.kind == "heading_only_tracking"
        ):
            # One bounded decision per stationary episode.  A rejected attempt
            # stays rejected until motion begins a new episode.
            stopped_episode_applied = True
    if not attempts:
        raise ValueError("no map-correction attempt could be formed")
    runtimes = np.asarray([row["end_to_end_runtime_ms"] for row in attempts])
    report = {
        "schema": "g1_structural_map_correction_replay_v1",
        "status": "pass" if packets else "fail",
        "command_capability": "structurally_unavailable",
        "motive_online_input": False,
        "map_path": str(args.map),
        "map_key": args.map_key,
        "map_digest": engine.map_digest.hex(),
        "initialization_mode": (
            "ui_pin_locked" if initialization is not None else "automatic_global_search"
        ),
        "treatment": args.treatment,
        "treatment_contract": {
            "correction_period_sec": args.correction_period_sec,
            "stationary_window_sec": args.stationary_window_sec,
            "stationary_linear_speed_mps": args.stationary_linear_speed_mps,
            "stationary_yaw_rate_radps": args.stationary_yaw_rate_radps,
            "motive_online_input": False,
        },
        "initialization_receipt": (
            None
            if args.initialization_receipt is None
            else str(args.initialization_receipt)
        ),
        "local_source_epoch": args.local_source_epoch,
        "attempt_count": len(attempts),
        "accepted_count": len(packets),
        "rejected_count": len(attempts) - len(packets),
        "end_to_end_runtime_ms": {
            "p50": float(np.quantile(runtimes, 0.50)),
            "p95": float(np.quantile(runtimes, 0.95)),
            "p99": float(np.quantile(runtimes, 0.99)),
            "maximum": float(np.max(runtimes)),
        },
        "maximum_input_age_ms": config.maximum_input_age_ns * 1e-6,
        "attempts": attempts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.packets_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.packets_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in packets),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "packets_output": str(args.packets_output),
                "status": report["status"],
                "attempt_count": report["attempt_count"],
                "accepted_count": report["accepted_count"],
                "runtime_p95_ms": report["end_to_end_runtime_ms"]["p95"],
            },
            sort_keys=True,
        )
    )
    return 0 if packets else 1


if __name__ == "__main__":
    raise SystemExit(main())
