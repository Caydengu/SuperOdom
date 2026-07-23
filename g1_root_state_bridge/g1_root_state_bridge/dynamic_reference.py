"""Fail-closed scoring against an independent timestamped pelvis reference.

The scorer deliberately separates three operations:

1. map the reference clock into the estimator clock with an explicitly
   validated affine mapping;
2. estimate one unit-scale rigid world-frame alignment from a designated
   calibration interval;
3. score a disjoint motion interval and a waist-only invariance interval.

It never estimates scale, retimes samples implicitly, or treats invalid
tracking as a usable pose.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


_VECTOR_LENGTH = 3
_QUATERNION_LENGTH = 4


def _finite_tuple(
    value: Sequence[float],
    *,
    length: int,
    field_name: str,
) -> tuple[float, ...]:
    if len(value) != length:
        raise ValueError(f"{field_name} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field_name} must contain only finite values")
    return result


@dataclass(frozen=True)
class PoseSample:
    """One pose sample with an explicit clock, frame, and calibration identity."""

    sequence: int
    time_ns: int
    clock_id: str
    frame_id: str
    calibration_id: str
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float] | None = None
    angular_velocity: tuple[float, float, float] | None = None
    valid: bool = True
    tracking_error_m: float | None = None

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        if self.time_ns < 0:
            raise ValueError("time_ns must be non-negative")
        if not self.clock_id:
            raise ValueError("clock_id must be non-empty")
        if not self.frame_id:
            raise ValueError("frame_id must be non-empty")
        if not self.calibration_id:
            raise ValueError("calibration_id must be non-empty")

        position = _finite_tuple(
            self.position,
            length=_VECTOR_LENGTH,
            field_name="position",
        )
        quaternion = _finite_tuple(
            self.quaternion_wxyz,
            length=_QUATERNION_LENGTH,
            field_name="quaternion_wxyz",
        )
        quaternion_norm = math.sqrt(sum(component * component for component in quaternion))
        if abs(quaternion_norm - 1.0) > 1e-3:
            raise ValueError("quaternion_wxyz must be unit length within 1e-3")

        linear_velocity = None
        if self.linear_velocity is not None:
            linear_velocity = _finite_tuple(
                self.linear_velocity,
                length=_VECTOR_LENGTH,
                field_name="linear_velocity",
            )
        angular_velocity = None
        if self.angular_velocity is not None:
            angular_velocity = _finite_tuple(
                self.angular_velocity,
                length=_VECTOR_LENGTH,
                field_name="angular_velocity",
            )
        tracking_error_m = self.tracking_error_m
        if tracking_error_m is not None:
            tracking_error_m = float(tracking_error_m)
            if not math.isfinite(tracking_error_m) or tracking_error_m < 0.0:
                raise ValueError(
                    "tracking_error_m must be finite and non-negative"
                )

        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion_wxyz", quaternion)
        object.__setattr__(self, "linear_velocity", linear_velocity)
        object.__setattr__(self, "angular_velocity", angular_velocity)
        object.__setattr__(self, "tracking_error_m", tracking_error_m)

    def as_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "time_ns": self.time_ns,
            "clock_id": self.clock_id,
            "frame_id": self.frame_id,
            "calibration_id": self.calibration_id,
            "position": self.position,
            "quaternion_wxyz": self.quaternion_wxyz,
            "linear_velocity": self.linear_velocity,
            "angular_velocity": self.angular_velocity,
            "valid": self.valid,
            "tracking_error_m": self.tracking_error_m,
        }


@dataclass(frozen=True)
class ClockMapping:
    """Validated affine mapping from a source clock into the target clock."""

    source_clock_id: str
    target_clock_id: str
    scale: float
    offset_ns: int
    residual_p95_ns: int
    calibration_id: str

    def __post_init__(self) -> None:
        if not self.source_clock_id or not self.target_clock_id:
            raise ValueError("clock identifiers must be non-empty")
        if self.source_clock_id == self.target_clock_id:
            raise ValueError("clock mapping must connect two different clocks")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("clock mapping scale must be finite and positive")
        if self.residual_p95_ns < 0:
            raise ValueError("clock mapping residual must be non-negative")
        if not self.calibration_id:
            raise ValueError("clock mapping calibration_id must be non-empty")

    def map_time_ns(self, source_time_ns: int) -> int:
        return int(round(self.scale * source_time_ns + self.offset_ns))


@dataclass(frozen=True)
class DynamicReferenceConfig:
    target_clock_id: str
    reference_calibration_id: str
    calibration_interval_ns: tuple[int, int]
    evaluation_interval_ns: tuple[int, int]
    waist_interval_ns: tuple[int, int]
    max_interpolation_gap_ns: int
    max_clock_mapping_residual_ns: int
    minimum_coverage: float
    horizontal_p95_limit_m: float
    vertical_p95_limit_m: float
    yaw_p95_limit_deg: float
    waist_position_residual_limit_m: float
    waist_yaw_residual_limit_deg: float

    def __post_init__(self) -> None:
        if not self.target_clock_id or not self.reference_calibration_id:
            raise ValueError("clock and calibration identifiers must be non-empty")
        for name in (
            "calibration_interval_ns",
            "evaluation_interval_ns",
            "waist_interval_ns",
        ):
            interval = getattr(self, name)
            if len(interval) != 2 or interval[0] < 0 or interval[1] <= interval[0]:
                raise ValueError(f"{name} must be an increasing non-negative interval")
        if self.max_interpolation_gap_ns <= 0:
            raise ValueError("max_interpolation_gap_ns must be positive")
        if self.max_clock_mapping_residual_ns < 0:
            raise ValueError("max_clock_mapping_residual_ns must be non-negative")
        if not 0.0 < self.minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must be in (0, 1]")
        for name in (
            "horizontal_p95_limit_m",
            "vertical_p95_limit_m",
            "yaw_p95_limit_deg",
            "waist_position_residual_limit_m",
            "waist_yaw_residual_limit_deg",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class DynamicReferenceReport:
    valid: bool
    gates_pass: bool
    invalid_reasons: tuple[str, ...]
    metrics: Mapping[str, float | int]
    gate_results: Mapping[str, bool]
    alignment_rotation: tuple[tuple[float, float, float], ...] | None = None
    alignment_translation: tuple[float, float, float] | None = None
    alignment_scale: float = 1.0


@dataclass(frozen=True)
class _ReferencePose:
    time_ns: int
    position: np.ndarray
    rotation: np.ndarray
    linear_velocity: np.ndarray | None
    angular_velocity: np.ndarray | None


@dataclass(frozen=True)
class _MatchedPose:
    estimator: PoseSample
    reference: _ReferencePose


def _quaternion_to_rotation(quaternion_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = quaternion_wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = (component / norm for component in (w, x, y, z))
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _slerp(
    first_wxyz: Sequence[float],
    second_wxyz: Sequence[float],
    fraction: float,
) -> tuple[float, float, float, float]:
    first = np.asarray(first_wxyz, dtype=np.float64)
    second = np.asarray(second_wxyz, dtype=np.float64)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = first + fraction * (second - first)
        result /= np.linalg.norm(result)
        return tuple(float(value) for value in result)
    angle = math.acos(dot)
    sine = math.sin(angle)
    result = (
        math.sin((1.0 - fraction) * angle) / sine * first
        + math.sin(fraction * angle) / sine * second
    )
    result /= np.linalg.norm(result)
    return tuple(float(value) for value in result)


def _strictly_increasing(samples: Sequence[PoseSample], attribute: str) -> bool:
    values = [getattr(sample, attribute) for sample in samples]
    return all(previous < current for previous, current in zip(values, values[1:]))


def _single_string_value(
    samples: Sequence[PoseSample],
    attribute: str,
) -> str | None:
    values = {str(getattr(sample, attribute)) for sample in samples}
    if len(values) != 1:
        return None
    return values.pop()


def _reference_pose(
    sample: PoseSample,
    *,
    mapped_time_ns: int,
) -> _ReferencePose:
    return _ReferencePose(
        time_ns=mapped_time_ns,
        position=np.asarray(sample.position, dtype=np.float64),
        rotation=_quaternion_to_rotation(sample.quaternion_wxyz),
        linear_velocity=(
            None
            if sample.linear_velocity is None
            else np.asarray(sample.linear_velocity, dtype=np.float64)
        ),
        angular_velocity=(
            None
            if sample.angular_velocity is None
            else np.asarray(sample.angular_velocity, dtype=np.float64)
        ),
    )


def _interpolate_reference(
    query_time_ns: int,
    reference: Sequence[tuple[int, PoseSample]],
    reference_times: Sequence[int],
    *,
    max_gap_ns: int,
) -> _ReferencePose | None:
    index = bisect_left(reference_times, query_time_ns)
    if index < len(reference_times) and reference_times[index] == query_time_ns:
        mapped_time_ns, sample = reference[index]
        return _reference_pose(sample, mapped_time_ns=mapped_time_ns)
    if index == 0 or index >= len(reference_times):
        return None
    left_time, left = reference[index - 1]
    right_time, right = reference[index]
    gap_ns = right_time - left_time
    if gap_ns <= 0 or gap_ns > max_gap_ns:
        return None
    fraction = (query_time_ns - left_time) / gap_ns
    position = (
        (1.0 - fraction) * np.asarray(left.position, dtype=np.float64)
        + fraction * np.asarray(right.position, dtype=np.float64)
    )
    quaternion = _slerp(left.quaternion_wxyz, right.quaternion_wxyz, fraction)

    linear_velocity = None
    if left.linear_velocity is not None and right.linear_velocity is not None:
        linear_velocity = (
            (1.0 - fraction) * np.asarray(left.linear_velocity, dtype=np.float64)
            + fraction * np.asarray(right.linear_velocity, dtype=np.float64)
        )
    angular_velocity = None
    if left.angular_velocity is not None and right.angular_velocity is not None:
        angular_velocity = (
            (1.0 - fraction) * np.asarray(left.angular_velocity, dtype=np.float64)
            + fraction * np.asarray(right.angular_velocity, dtype=np.float64)
        )
    return _ReferencePose(
        time_ns=query_time_ns,
        position=position,
        rotation=_quaternion_to_rotation(quaternion),
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
    )


def _match_interval(
    estimator: Sequence[PoseSample],
    reference: Sequence[tuple[int, PoseSample]],
    *,
    interval_ns: tuple[int, int],
    max_gap_ns: int,
) -> tuple[list[_MatchedPose], int]:
    start_ns, end_ns = interval_ns
    candidates = [
        sample for sample in estimator if start_ns <= sample.time_ns <= end_ns
    ]
    valid_reference = [
        (time_ns, sample) for time_ns, sample in reference if sample.valid
    ]
    reference_times = [time_ns for time_ns, _ in valid_reference]
    matches: list[_MatchedPose] = []
    for sample in candidates:
        if not sample.valid:
            continue
        interpolated = _interpolate_reference(
            sample.time_ns,
            valid_reference,
            reference_times,
            max_gap_ns=max_gap_ns,
        )
        if interpolated is not None:
            matches.append(_MatchedPose(sample, interpolated))
    return matches, len(candidates)


def _alignment_from_matches(
    matches: Sequence[_MatchedPose],
) -> tuple[np.ndarray, np.ndarray]:
    relative_rotations = np.stack(
        [
            _quaternion_to_rotation(match.estimator.quaternion_wxyz)
            @ match.reference.rotation.T
            for match in matches
        ]
    )
    rotation_sum = np.sum(relative_rotations, axis=0)
    left, _, right_transpose = np.linalg.svd(rotation_sum)
    alignment_rotation = left @ right_transpose
    if np.linalg.det(alignment_rotation) < 0.0:
        left[:, -1] *= -1.0
        alignment_rotation = left @ right_transpose
    translations = np.stack(
        [
            np.asarray(match.estimator.position, dtype=np.float64)
            - alignment_rotation @ match.reference.position
            for match in matches
        ]
    )
    return alignment_rotation, np.mean(translations, axis=0)


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _yaw_from_rotation(rotation: np.ndarray) -> float:
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _invalid_report(
    reasons: Iterable[str],
    *,
    metrics: Mapping[str, float | int] | None = None,
) -> DynamicReferenceReport:
    return DynamicReferenceReport(
        valid=False,
        gates_pass=False,
        invalid_reasons=tuple(dict.fromkeys(reasons)),
        metrics={} if metrics is None else dict(metrics),
        gate_results={},
    )


def score_dynamic_reference(
    estimator_samples: Iterable[PoseSample],
    reference_samples: Iterable[PoseSample],
    *,
    config: DynamicReferenceConfig,
    clock_mapping: ClockMapping | None = None,
) -> DynamicReferenceReport:
    """Score estimator pelvis samples against independent reference samples.

    A report can be scientifically valid while failing one or more accuracy
    gates. Invalid evidence has ``valid=False`` and never passes gates.
    """

    estimator = list(estimator_samples)
    reference = list(reference_samples)
    reasons: list[str] = []
    if len(estimator) < 2:
        reasons.append("insufficient_estimator_samples")
    if len(reference) < 2:
        reasons.append("insufficient_reference_samples")
    if reasons:
        return _invalid_report(reasons)

    if not _strictly_increasing(estimator, "time_ns"):
        reasons.append("estimator_time_not_strictly_increasing")
    if not _strictly_increasing(reference, "time_ns"):
        reasons.append("reference_time_not_strictly_increasing")
    if not _strictly_increasing(estimator, "sequence"):
        reasons.append("estimator_sequence_not_strictly_increasing")
    if not _strictly_increasing(reference, "sequence"):
        reasons.append("reference_sequence_not_strictly_increasing")

    estimator_clock = _single_string_value(estimator, "clock_id")
    reference_clock = _single_string_value(reference, "clock_id")
    if estimator_clock is None:
        reasons.append("estimator_clock_not_unique")
    elif estimator_clock != config.target_clock_id:
        reasons.append("estimator_clock_not_target")
    if reference_clock is None:
        reasons.append("reference_clock_not_unique")

    if _single_string_value(estimator, "frame_id") is None:
        reasons.append("estimator_frame_not_unique")
    if _single_string_value(reference, "frame_id") is None:
        reasons.append("reference_frame_not_unique")
    reference_calibration = _single_string_value(reference, "calibration_id")
    if reference_calibration != config.reference_calibration_id:
        reasons.append("reference_calibration_not_allowed")

    if reference_clock is not None and reference_clock != config.target_clock_id:
        if clock_mapping is None:
            reasons.append("clock_mapping_required")
        else:
            if (
                clock_mapping.source_clock_id != reference_clock
                or clock_mapping.target_clock_id != config.target_clock_id
            ):
                reasons.append("clock_mapping_identity_mismatch")
            if (
                clock_mapping.residual_p95_ns
                > config.max_clock_mapping_residual_ns
            ):
                reasons.append("clock_mapping_residual_exceeds_limit")
    elif clock_mapping is not None:
        reasons.append("clock_mapping_unexpected_for_common_clock")

    if reasons:
        return _invalid_report(reasons)

    assert reference_clock is not None
    mapped_reference: list[tuple[int, PoseSample]] = []
    for sample in reference:
        mapped_time_ns = sample.time_ns
        if reference_clock != config.target_clock_id:
            assert clock_mapping is not None
            mapped_time_ns = clock_mapping.map_time_ns(sample.time_ns)
        if mapped_time_ns < 0:
            return _invalid_report(("mapped_reference_time_negative",))
        mapped_reference.append((mapped_time_ns, sample))
    if not all(
        previous[0] < current[0]
        for previous, current in zip(mapped_reference, mapped_reference[1:])
    ):
        return _invalid_report(("mapped_reference_time_not_strictly_increasing",))

    calibration_matches, calibration_candidates = _match_interval(
        estimator,
        mapped_reference,
        interval_ns=config.calibration_interval_ns,
        max_gap_ns=config.max_interpolation_gap_ns,
    )
    evaluation_matches, evaluation_candidates = _match_interval(
        estimator,
        mapped_reference,
        interval_ns=config.evaluation_interval_ns,
        max_gap_ns=config.max_interpolation_gap_ns,
    )
    waist_matches, waist_candidates = _match_interval(
        estimator,
        mapped_reference,
        interval_ns=config.waist_interval_ns,
        max_gap_ns=config.max_interpolation_gap_ns,
    )
    metrics: dict[str, float | int] = {
        "calibration_candidate_count": calibration_candidates,
        "calibration_match_count": len(calibration_matches),
        "evaluation_candidate_count": evaluation_candidates,
        "evaluation_match_count": len(evaluation_matches),
        "waist_candidate_count": waist_candidates,
        "waist_match_count": len(waist_matches),
        "calibration_coverage": (
            len(calibration_matches) / calibration_candidates
            if calibration_candidates
            else 0.0
        ),
        "evaluation_coverage": (
            len(evaluation_matches) / evaluation_candidates
            if evaluation_candidates
            else 0.0
        ),
        "waist_coverage": (
            len(waist_matches) / waist_candidates if waist_candidates else 0.0
        ),
    }

    if len(calibration_matches) < 3:
        reasons.append("insufficient_calibration_matches")
    if metrics["calibration_coverage"] < config.minimum_coverage:
        reasons.append("calibration_coverage_below_minimum")
    if len(evaluation_matches) < 3:
        reasons.append("insufficient_evaluation_matches")
    if metrics["evaluation_coverage"] < config.minimum_coverage:
        reasons.append("evaluation_coverage_below_minimum")
    if len(waist_matches) < 3:
        reasons.append("insufficient_waist_matches")
    if metrics["waist_coverage"] < config.minimum_coverage:
        reasons.append("waist_coverage_below_minimum")
    if reasons:
        return _invalid_report(reasons, metrics=metrics)

    alignment_rotation, alignment_translation = _alignment_from_matches(
        calibration_matches
    )

    horizontal_errors: list[float] = []
    vertical_errors: list[float] = []
    yaw_errors_deg: list[float] = []
    linear_velocity_errors: list[float] = []
    angular_velocity_errors: list[float] = []
    for match in evaluation_matches:
        aligned_position = (
            alignment_rotation @ match.reference.position + alignment_translation
        )
        position_error = (
            np.asarray(match.estimator.position, dtype=np.float64)
            - aligned_position
        )
        horizontal_errors.append(float(np.linalg.norm(position_error[:2])))
        vertical_errors.append(abs(float(position_error[2])))

        estimator_rotation = _quaternion_to_rotation(
            match.estimator.quaternion_wxyz
        )
        aligned_reference_rotation = alignment_rotation @ match.reference.rotation
        rotation_error = estimator_rotation @ aligned_reference_rotation.T
        yaw_errors_deg.append(
            abs(math.degrees(_wrap_angle(_yaw_from_rotation(rotation_error))))
        )

        if (
            match.estimator.linear_velocity is not None
            and match.reference.linear_velocity is not None
        ):
            reference_velocity = (
                alignment_rotation @ match.reference.linear_velocity
            )
            linear_velocity_errors.append(
                float(
                    np.linalg.norm(
                        np.asarray(
                            match.estimator.linear_velocity,
                            dtype=np.float64,
                        )
                        - reference_velocity
                    )
                )
            )
        if (
            match.estimator.angular_velocity is not None
            and match.reference.angular_velocity is not None
        ):
            reference_angular_velocity = (
                alignment_rotation @ match.reference.angular_velocity
            )
            angular_velocity_errors.append(
                float(
                    np.linalg.norm(
                        np.asarray(
                            match.estimator.angular_velocity,
                            dtype=np.float64,
                        )
                        - reference_angular_velocity
                    )
                )
            )

    metrics.update(
        {
            "horizontal_error_p50_m": _percentile(horizontal_errors, 50.0),
            "horizontal_error_p95_m": _percentile(horizontal_errors, 95.0),
            "horizontal_error_max_m": max(horizontal_errors),
            "vertical_error_p50_m": _percentile(vertical_errors, 50.0),
            "vertical_error_p95_m": _percentile(vertical_errors, 95.0),
            "vertical_error_max_m": max(vertical_errors),
            "yaw_error_p50_deg": _percentile(yaw_errors_deg, 50.0),
            "yaw_error_p95_deg": _percentile(yaw_errors_deg, 95.0),
            "yaw_error_max_deg": max(yaw_errors_deg),
            "linear_velocity_match_count": len(linear_velocity_errors),
            "angular_velocity_match_count": len(angular_velocity_errors),
        }
    )
    if linear_velocity_errors:
        metrics.update(
            {
                "linear_velocity_error_p50_mps": _percentile(
                    linear_velocity_errors, 50.0
                ),
                "linear_velocity_error_p95_mps": _percentile(
                    linear_velocity_errors, 95.0
                ),
                "linear_velocity_error_max_mps": max(linear_velocity_errors),
            }
        )
    if angular_velocity_errors:
        metrics.update(
            {
                "angular_velocity_error_p50_radps": _percentile(
                    angular_velocity_errors, 50.0
                ),
                "angular_velocity_error_p95_radps": _percentile(
                    angular_velocity_errors, 95.0
                ),
                "angular_velocity_error_max_radps": max(angular_velocity_errors),
            }
        )

    waist_position_errors: list[np.ndarray] = []
    waist_yaw_errors: list[float] = []
    for match in waist_matches:
        aligned_position = (
            alignment_rotation @ match.reference.position + alignment_translation
        )
        waist_position_errors.append(
            np.asarray(match.estimator.position, dtype=np.float64)
            - aligned_position
        )
        estimator_rotation = _quaternion_to_rotation(
            match.estimator.quaternion_wxyz
        )
        aligned_reference_rotation = alignment_rotation @ match.reference.rotation
        waist_yaw_errors.append(
            _wrap_angle(
                _yaw_from_rotation(
                    estimator_rotation @ aligned_reference_rotation.T
                )
            )
        )
    initial_position_error = waist_position_errors[0]
    initial_yaw_error = waist_yaw_errors[0]
    waist_position_residuals = [
        float(np.linalg.norm(error - initial_position_error))
        for error in waist_position_errors
    ]
    waist_yaw_residuals_deg = [
        abs(math.degrees(_wrap_angle(error - initial_yaw_error)))
        for error in waist_yaw_errors
    ]
    metrics["waist_position_residual_max_m"] = max(waist_position_residuals)
    metrics["waist_yaw_residual_max_deg"] = max(waist_yaw_residuals_deg)

    gate_results = {
        "horizontal_p95": (
            metrics["horizontal_error_p95_m"] <= config.horizontal_p95_limit_m
        ),
        "vertical_p95": (
            metrics["vertical_error_p95_m"] <= config.vertical_p95_limit_m
        ),
        "yaw_p95": metrics["yaw_error_p95_deg"] <= config.yaw_p95_limit_deg,
        "waist_position_residual": (
            metrics["waist_position_residual_max_m"]
            <= config.waist_position_residual_limit_m
        ),
        "waist_yaw_residual": (
            metrics["waist_yaw_residual_max_deg"]
            <= config.waist_yaw_residual_limit_deg
        ),
    }
    return DynamicReferenceReport(
        valid=True,
        gates_pass=all(gate_results.values()),
        invalid_reasons=(),
        metrics=metrics,
        gate_results=gate_results,
        alignment_rotation=tuple(
            tuple(float(value) for value in row) for row in alignment_rotation
        ),
        alignment_translation=tuple(
            float(value) for value in alignment_translation
        ),
        alignment_scale=1.0,
    )
