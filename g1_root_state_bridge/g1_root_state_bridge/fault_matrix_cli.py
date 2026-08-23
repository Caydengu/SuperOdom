"""Run socket-free hard-failure injections against the selected local lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from g1_root_state_bridge.clock_sync import ClockMapConfig, OnlineAffineClockMapper
from g1_root_state_bridge.joint_contract import CANONICAL_G1_JOINT_NAMES, TimedJointSample
from g1_root_state_bridge.kiss_registration import RegistrationResult
from g1_root_state_bridge.live_pipeline import (
    LiveLocalizationConfig,
    LiveLocalizationError,
    SelectedLocalizationPipeline,
)
from g1_root_state_bridge.protocol import REQUIRED_ROOT_FUSION_FLAGS


class _DeterministicRegistration:
    def __init__(self) -> None:
        self.index = 0

    def register(self, _points: np.ndarray) -> RegistrationResult:
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = 0.01 * self.index
        self.index += 1
        return RegistrationResult(pose, 0.1, 0.5)

    def reset(self) -> None:
        self.index = 0


def _joint(time_ns: int, sequence: int) -> TimedJointSample:
    return TimedJointSample(
        stamp_ns=time_ns,
        receipt_ns=time_ns,
        names=CANONICAL_G1_JOINT_NAMES,
        position=(0.0,) * 29,
        velocity=(0.0,) * 29,
        sequence=sequence,
    )


def _pipeline(*, joint_period_ns: int = 5_000_000) -> tuple[SelectedLocalizationPipeline, int]:
    pipeline = SelectedLocalizationPipeline(
        _DeterministicRegistration(),
        LiveLocalizationConfig(gyro_bias_radps=(0.0, 0.0, 0.0)),
    )
    start_ns = 1_000_000_000
    for index in range(40):
        stamp_ns = start_ns - 20_000_000 + index * 5_000_000
        pipeline.append_imu(stamp_ns, np.zeros(3, dtype=np.float64))
    for index in range(40):
        stamp_ns = start_ns - 15_000_000 + index * joint_period_ns
        pipeline.append_joint(_joint(stamp_ns, index))
    return pipeline, start_ns


def _scan(pipeline: SelectedLocalizationPipeline, start_ns: int, **kwargs: object):
    return pipeline.process_scan(
        np.tile(np.asarray(((1.0, 0.0, 0.0),)), (20, 1)),
        np.linspace(0.0, 0.1, 20),
        scan_start_time_ns=start_ns,
        publish_time_ns=start_ns + 150_000_000,
        **kwargs,
    )


def run_fault_matrix() -> dict[str, object]:
    rows: list[dict[str, object]] = []

    for name, mutation in (
        ("clock_invalid", {"clock_valid": False}),
        ("calibration_invalid", {"calibration_valid": False}),
    ):
        pipeline, start_ns = _pipeline()
        output = _scan(pipeline, start_ns, **mutation)
        fused_healthy = (
            output.packet.health_flags & REQUIRED_ROOT_FUSION_FLAGS
        ) == REQUIRED_ROOT_FUSION_FLAGS
        rows.append(
            {
                "fault": name,
                "output_emitted": True,
                "fused_healthy": fused_healthy,
                "false_healthy": fused_healthy,
            }
        )

    pipeline, start_ns = _pipeline(joint_period_ns=30_000_000)
    try:
        _scan(pipeline, start_ns)
    except LiveLocalizationError as error:
        rows.append(
            {
                "fault": "joint_bracket_stale",
                "output_emitted": False,
                "fused_healthy": False,
                "false_healthy": False,
                "reason": str(error),
            }
        )
    else:
        rows.append(
            {
                "fault": "joint_bracket_stale",
                "output_emitted": True,
                "fused_healthy": True,
                "false_healthy": True,
            }
        )

    pipeline = SelectedLocalizationPipeline(
        _DeterministicRegistration(),
        LiveLocalizationConfig(gyro_bias_radps=(0.0, 0.0, 0.0)),
    )
    try:
        _scan(pipeline, 1_000_000_000)
    except LiveLocalizationError as error:
        rows.append(
            {
                "fault": "imu_coverage_missing",
                "output_emitted": False,
                "fused_healthy": False,
                "false_healthy": False,
                "reason": str(error),
            }
        )
    else:
        rows.append(
            {
                "fault": "imu_coverage_missing",
                "output_emitted": True,
                "fused_healthy": True,
                "false_healthy": True,
            }
        )

    mapper = OnlineAffineClockMapper(
        ClockMapConfig(
            minimum_samples=3,
            minimum_span_ns=1,
            refit_period_samples=1,
        )
    )
    for index in range(3):
        mapper.observe(10_000 + index * 100, 20_000 + index * 100)
    regression = mapper.observe(5_000, 21_000)
    rows.append(
        {
            "fault": "source_clock_regression",
            "output_emitted": False,
            "fused_healthy": False,
            "false_healthy": regression.valid,
            "new_source_epoch": regression.source_epoch,
            "clock_valid": regression.valid,
        }
    )

    pipeline, start_ns = _pipeline()
    before = _scan(pipeline, start_ns)
    pipeline.reset(1)
    for index in range(40):
        stamp_ns = start_ns - 20_000_000 + index * 5_000_000
        pipeline.append_imu(stamp_ns, np.zeros(3, dtype=np.float64))
        pipeline.append_joint(_joint(stamp_ns, index))
    after = _scan(pipeline, start_ns)
    reset_valid = (
        before.packet.sequence == 1
        and after.packet.sequence == 1
        and after.packet.source_epoch == 1
    )
    rows.append(
        {
            "fault": "source_epoch_reset",
            "output_emitted": True,
            "fused_healthy": reset_valid,
            "false_healthy": False,
            "recovery_valid": reset_valid,
        }
    )

    false_healthy = sum(bool(row["false_healthy"]) for row in rows)
    return {
        "schema": "g1_kiss_live_localization_fault_matrix_v1",
        "status": "pass" if false_healthy == 0 else "fail",
        "fault_count": len(rows),
        "false_healthy_count": false_healthy,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = run_fault_matrix()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
