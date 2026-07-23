from __future__ import annotations

import math

import pytest

from g1_root_state_bridge.dynamic_reference import (
    ClockMapping,
    DynamicReferenceConfig,
    PoseSample,
    score_dynamic_reference,
)


NS = 1_000_000_000


def _yaw_quaternion(yaw_rad: float) -> tuple[float, float, float, float]:
    return (
        math.cos(yaw_rad / 2.0),
        0.0,
        0.0,
        math.sin(yaw_rad / 2.0),
    )


def _trajectory(
    *,
    clock_id: str = "oslo_monotonic",
    time_offset_ns: int = 0,
    frame_id: str = "world",
    calibration_id: str = "reference-calibration-v1",
) -> list[PoseSample]:
    samples: list[PoseSample] = []
    for index in range(61):
        time_s = index * 0.1
        if time_s <= 1.0:
            x = 0.0
            yaw = 0.0
            vx = 0.0
            wz = 0.0
        elif time_s <= 3.0:
            x = 0.1 * (time_s - 1.0)
            yaw = 0.05 * (time_s - 1.0)
            vx = 0.1
            wz = 0.05
        else:
            x = 0.2
            yaw = 0.1
            vx = 0.0
            wz = 0.0
        samples.append(
            PoseSample(
                sequence=index,
                time_ns=index * NS // 10 + time_offset_ns,
                clock_id=clock_id,
                frame_id=frame_id,
                calibration_id=calibration_id,
                position=(x, 0.0, 0.8),
                quaternion_wxyz=_yaw_quaternion(yaw),
                linear_velocity=(vx, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, wz),
            )
        )
    return samples


def _config(**changes: object) -> DynamicReferenceConfig:
    values: dict[str, object] = {
        "target_clock_id": "oslo_monotonic",
        "reference_calibration_id": "reference-calibration-v1",
        "calibration_interval_ns": (0, NS),
        "evaluation_interval_ns": (NS, 3 * NS),
        "waist_interval_ns": (4 * NS, 6 * NS),
        "max_interpolation_gap_ns": 150_000_000,
        "max_clock_mapping_residual_ns": 5_000_000,
        "minimum_coverage": 0.9,
        "horizontal_p95_limit_m": 0.05,
        "vertical_p95_limit_m": 0.03,
        "yaw_p95_limit_deg": 1.5,
        "waist_position_residual_limit_m": 0.02,
        "waist_yaw_residual_limit_deg": 0.5,
    }
    values.update(changes)
    return DynamicReferenceConfig(**values)


def test_aligned_identity_trajectory_passes_all_frozen_gates() -> None:
    reference = _trajectory(frame_id="mocap_world")
    estimator = _trajectory(frame_id="superodom_world")

    report = score_dynamic_reference(estimator, reference, config=_config())

    assert report.valid is True
    assert report.gates_pass is True
    assert report.invalid_reasons == ()
    assert report.metrics["evaluation_coverage"] == pytest.approx(1.0)
    assert report.metrics["horizontal_error_p95_m"] < 1e-9
    assert report.metrics["vertical_error_p95_m"] < 1e-9
    assert report.metrics["yaw_error_p95_deg"] < 1e-9
    assert report.metrics["waist_position_residual_max_m"] < 1e-9
    assert report.metrics["waist_yaw_residual_max_deg"] < 1e-9
    assert report.metrics["linear_velocity_error_p95_mps"] < 1e-9


def test_stationary_alignment_removes_only_a_fixed_rigid_world_transform() -> None:
    reference = _trajectory(frame_id="mocap_world")
    transform_yaw = math.radians(35.0)
    c = math.cos(transform_yaw)
    s = math.sin(transform_yaw)
    estimator: list[PoseSample] = []
    for sample in reference:
        x, y, z = sample.position
        vx, vy, vz = sample.linear_velocity or (0.0, 0.0, 0.0)
        yaw = 2.0 * math.atan2(
            sample.quaternion_wxyz[3], sample.quaternion_wxyz[0]
        )
        estimator.append(
            PoseSample(
                sequence=sample.sequence,
                time_ns=sample.time_ns,
                clock_id=sample.clock_id,
                frame_id="superodom_world",
                calibration_id=sample.calibration_id,
                position=(c * x - s * y + 1.2, s * x + c * y - 0.4, z + 0.3),
                quaternion_wxyz=_yaw_quaternion(yaw + transform_yaw),
                linear_velocity=(c * vx - s * vy, s * vx + c * vy, vz),
                angular_velocity=sample.angular_velocity,
            )
        )

    report = score_dynamic_reference(estimator, reference, config=_config())

    assert report.valid is True
    assert report.gates_pass is True
    assert report.metrics["horizontal_error_p95_m"] < 1e-9
    assert report.metrics["vertical_error_p95_m"] < 1e-9
    assert report.metrics["yaw_error_p95_deg"] < 1e-9
    assert report.alignment_scale == 1.0


def test_six_centimeter_dynamic_horizontal_error_fails_pose_gate() -> None:
    reference = _trajectory()
    estimator = []
    for sample in _trajectory():
        position = sample.position
        if NS < sample.time_ns <= 3 * NS:
            position = (position[0] + 0.06, position[1], position[2])
        estimator.append(
            PoseSample(
                **{
                    **sample.as_dict(),
                    "position": position,
                }
            )
        )

    report = score_dynamic_reference(estimator, reference, config=_config())

    assert report.valid is True
    assert report.gates_pass is False
    assert report.gate_results["horizontal_p95"] is False
    assert report.metrics["horizontal_error_p95_m"] == pytest.approx(0.06)


def test_different_clock_requires_a_validated_affine_mapping() -> None:
    reference = _trajectory(
        clock_id="motive_software",
        time_offset_ns=25 * NS,
    )
    estimator = _trajectory()

    missing = score_dynamic_reference(estimator, reference, config=_config())
    assert missing.valid is False
    assert "clock_mapping_required" in missing.invalid_reasons

    mapping = ClockMapping(
        source_clock_id="motive_software",
        target_clock_id="oslo_monotonic",
        scale=1.0,
        offset_ns=-25 * NS,
        residual_p95_ns=1_000_000,
        calibration_id="motive-to-oslo-2026-07-22",
    )
    mapped = score_dynamic_reference(
        estimator,
        reference,
        config=_config(),
        clock_mapping=mapping,
    )
    assert mapped.valid is True
    assert mapped.gates_pass is True


def test_excessive_clock_fit_residual_invalidates_evidence() -> None:
    reference = _trajectory(
        clock_id="motive_software",
        time_offset_ns=25 * NS,
    )
    mapping = ClockMapping(
        source_clock_id="motive_software",
        target_clock_id="oslo_monotonic",
        scale=1.0,
        offset_ns=-25 * NS,
        residual_p95_ns=6_000_000,
        calibration_id="bad-clock-fit",
    )

    report = score_dynamic_reference(
        _trajectory(),
        reference,
        config=_config(),
        clock_mapping=mapping,
    )

    assert report.valid is False
    assert "clock_mapping_residual_exceeds_limit" in report.invalid_reasons


def test_nonmonotonic_time_and_unknown_calibration_fail_closed() -> None:
    reference = _trajectory()
    reference[20], reference[21] = reference[21], reference[20]
    nonmonotonic = score_dynamic_reference(
        _trajectory(), reference, config=_config()
    )
    assert nonmonotonic.valid is False
    assert "reference_time_not_strictly_increasing" in nonmonotonic.invalid_reasons

    wrong_calibration = _trajectory(calibration_id="unknown")
    unknown = score_dynamic_reference(
        _trajectory(), wrong_calibration, config=_config()
    )
    assert unknown.valid is False
    assert "reference_calibration_not_allowed" in unknown.invalid_reasons


def test_tracking_loss_and_large_gaps_reduce_coverage_and_invalidate_trace() -> None:
    reference = _trajectory()
    reference = [
        PoseSample(**{**sample.as_dict(), "valid": not (12 <= sample.sequence <= 27)})
        for sample in reference
    ]

    report = score_dynamic_reference(_trajectory(), reference, config=_config())

    assert report.valid is False
    assert "evaluation_coverage_below_minimum" in report.invalid_reasons
    assert report.metrics["evaluation_coverage"] < 0.9


def test_waist_only_three_centimeter_residual_fails_invariance_gate() -> None:
    reference = _trajectory()
    estimator = []
    for sample in _trajectory():
        position = sample.position
        quaternion = sample.quaternion_wxyz
        if sample.time_ns >= 4 * NS:
            fraction = (sample.time_ns - 4 * NS) / (2 * NS)
            position = (position[0] + 0.03 * fraction, position[1], position[2])
            quaternion = _yaw_quaternion(0.1 + math.radians(0.8) * fraction)
        estimator.append(
            PoseSample(
                **{
                    **sample.as_dict(),
                    "position": position,
                    "quaternion_wxyz": quaternion,
                }
            )
        )

    report = score_dynamic_reference(estimator, reference, config=_config())

    assert report.valid is True
    assert report.gates_pass is False
    assert report.gate_results["waist_position_residual"] is False
    assert report.gate_results["waist_yaw_residual"] is False
    assert report.metrics["waist_position_residual_max_m"] == pytest.approx(0.03)
    assert report.metrics["waist_yaw_residual_max_deg"] == pytest.approx(0.8)

