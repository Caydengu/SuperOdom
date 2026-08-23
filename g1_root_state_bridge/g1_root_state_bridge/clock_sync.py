"""Online affine source-clock admission for live G1 localization streams."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class ClockSynchronizationError(ValueError):
    """A clock observation violates the monotonic source/receipt contract."""


@dataclass(frozen=True)
class ClockMapConfig:
    minimum_samples: int = 8
    maximum_samples: int = 4_096
    minimum_span_ns: int = 500_000_000
    maximum_residual_p95_ns: int = 5_000_000
    maximum_transport_delay_ns: int = 10_000_000
    maximum_rate_error_ppm: float = 2_000.0
    lower_envelope_quantile: float = 0.0
    rate_fit_quantile: float = 0.5
    refit_period_samples: int = 32

    def __post_init__(self) -> None:
        if not 3 <= self.minimum_samples <= self.maximum_samples:
            raise ClockSynchronizationError("clock sample bounds are invalid")
        if self.maximum_residual_p95_ns <= 0:
            raise ClockSynchronizationError("clock residual limit must be positive")
        if self.maximum_transport_delay_ns <= 0:
            raise ClockSynchronizationError("clock transport-delay limit must be positive")
        if self.minimum_span_ns <= 0:
            raise ClockSynchronizationError("clock span limit must be positive")
        if not math.isfinite(self.maximum_rate_error_ppm) or self.maximum_rate_error_ppm <= 0:
            raise ClockSynchronizationError("clock rate limit must be finite and positive")
        if not 0.0 <= self.lower_envelope_quantile < 0.5:
            raise ClockSynchronizationError("lower-envelope quantile must be in [0,0.5)")
        if not 0.0 < self.rate_fit_quantile <= 1.0:
            raise ClockSynchronizationError("rate-fit quantile must be in (0,1]")
        if self.refit_period_samples <= 0:
            raise ClockSynchronizationError("clock refit period must be positive")


@dataclass(frozen=True)
class ClockMapEstimate:
    mapped_time_ns: int
    source_epoch: int
    valid: bool
    fit_valid: bool
    sample_valid: bool
    health_reasons: tuple[str, ...]
    sample_count: int
    slope: float
    residual_p95_ns: float
    transport_delay_ns: int


class ClockSampleDisposition(str, Enum):
    ACCEPT = "accept"
    DROP_SAMPLE = "drop_sample"
    RESET_LANE = "reset_lane"


def classify_clock_estimate(
    estimate: ClockMapEstimate, *, lane_active: bool
) -> ClockSampleDisposition:
    """Keep packet jitter separate from persistent clock-fit failure."""

    if not estimate.fit_valid:
        return (
            ClockSampleDisposition.RESET_LANE
            if lane_active
            else ClockSampleDisposition.DROP_SAMPLE
        )
    if not estimate.sample_valid:
        return ClockSampleDisposition.DROP_SAMPLE
    return ClockSampleDisposition.ACCEPT


@dataclass(frozen=True)
class AffineLowerEnvelopeClock:
    """One batch source-to-receipt mapping using the least-delayed envelope."""

    source_origin_ns: int
    receipt_origin_ns: int
    slope: float
    lower_envelope_residual_ns: float
    residual_p95_ns: float

    def map_time_ns(self, source_time_ns: int) -> int:
        return int(
            round(
                self.receipt_origin_ns
                + self.lower_envelope_residual_ns
                + self.slope * (source_time_ns - self.source_origin_ns)
            )
        )


def fit_affine_lower_envelope_clock(
    source_time_ns: np.ndarray,
    receipt_time_ns: np.ndarray,
    *,
    lower_quantile: float = 0.01,
    rate_fit_quantile: float = 1.0,
) -> AffineLowerEnvelopeClock:
    """Fit clock rate freely, then choose the least-delayed residual envelope."""

    source = np.asarray(source_time_ns, dtype=np.int64)
    receipt = np.asarray(receipt_time_ns, dtype=np.int64)
    if source.ndim != 1 or receipt.shape != source.shape or source.size < 2:
        raise ClockSynchronizationError(
            "clock fit requires equal one-dimensional arrays with at least two samples"
        )
    if not np.all(np.diff(source) > 0) or not np.all(np.diff(receipt) > 0):
        raise ClockSynchronizationError("clock fit samples must be strictly increasing")
    if not 0.0 <= lower_quantile < 0.5:
        raise ClockSynchronizationError("lower-envelope quantile must be in [0,0.5)")
    if not 0.0 < rate_fit_quantile <= 1.0:
        raise ClockSynchronizationError("rate-fit quantile must be in (0,1]")
    source_origin = int(source[0])
    receipt_origin = int(receipt[0])
    x = source.astype(np.float64) - source_origin
    y = receipt.astype(np.float64) - receipt_origin
    centered_x = x - np.mean(x)
    centered_y = y - np.mean(y)
    denominator = float(np.dot(centered_x, centered_x))
    if denominator <= 0.0:
        raise ClockSynchronizationError("source clock span is zero")
    slope = float(np.dot(centered_x, centered_y) / denominator)
    intercept = float(np.mean(y) - slope * np.mean(x))
    admitted: np.ndarray | None = None
    if rate_fit_quantile < 1.0 and source.size >= 4:
        first_residual = y - (intercept + slope * x)
        admitted = first_residual <= np.quantile(first_residual, rate_fit_quantile)
        admitted_x = x[admitted]
        admitted_y = y[admitted]
        centered_x = admitted_x - np.mean(admitted_x)
        centered_y = admitted_y - np.mean(admitted_y)
        denominator = float(np.dot(centered_x, centered_x))
        if denominator <= 0.0:
            raise ClockSynchronizationError("lower-delay clock fit has zero span")
        slope = float(np.dot(centered_x, centered_y) / denominator)
        intercept = float(np.mean(admitted_y) - slope * np.mean(admitted_x))
    residual = y - (intercept + slope * x)
    lower_residual = intercept + float(np.quantile(residual, lower_quantile))
    residual_for_health = residual if admitted is None else residual[admitted]
    centered_residual = residual_for_health - np.median(residual_for_health)
    residual_p95 = float(np.quantile(np.abs(centered_residual), 0.95))
    return AffineLowerEnvelopeClock(
        source_origin_ns=source_origin,
        receipt_origin_ns=receipt_origin,
        slope=slope,
        lower_envelope_residual_ns=lower_residual,
        residual_p95_ns=residual_p95,
    )


class OnlineAffineClockMapper:
    """Map one monotonic source clock into receiving-host realtime.

    The mapper never treats callback receipt time as source time.  Receipt time
    is only an observation used to estimate an affine mapping.  The mapping is
    unhealthy until enough monotonic pairs constrain its slope and residual.
    A source regression starts a new epoch and invalidates the fit.
    """

    def __init__(self, config: ClockMapConfig = ClockMapConfig()) -> None:
        self.config = config
        self._pairs: deque[tuple[int, int]] = deque(maxlen=config.maximum_samples)
        self._source_epoch = 0
        self._last_source_ns: int | None = None
        self._last_receipt_ns: int | None = None
        self._slope = 1.0
        self._offset_ns = 0.0
        self._residual_p95_ns = math.inf
        self._source_origin_ns = 0
        self._receipt_origin_ns = 0
        self._span_ns = 0
        self._mapping_causal = False
        self._last_mapped_observation_ns: int | None = None
        self._observations_since_fit = 0
        self._current_transport_delay_ns = math.inf

    @property
    def source_epoch(self) -> int:
        return self._source_epoch

    @property
    def last_source_time_ns(self) -> int | None:
        return self._last_source_ns

    @property
    def fit_valid(self) -> bool:
        rate_error_ppm = abs(self._slope - 1.0) * 1e6
        return (
            len(self._pairs) >= self.config.minimum_samples
            and self._span_ns >= self.config.minimum_span_ns
            and rate_error_ppm <= self.config.maximum_rate_error_ppm
            and self._residual_p95_ns <= self.config.maximum_residual_p95_ns
        )

    @property
    def sample_valid(self) -> bool:
        return (
            self._mapping_causal
            and 0
            <= self._current_transport_delay_ns
            <= self.config.maximum_transport_delay_ns
        )

    @property
    def valid(self) -> bool:
        return self.fit_valid and self.sample_valid

    @property
    def health_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if len(self._pairs) < self.config.minimum_samples:
            reasons.append("insufficient_samples")
        if self._span_ns < self.config.minimum_span_ns:
            reasons.append("insufficient_span")
        if abs(self._slope - 1.0) * 1e6 > self.config.maximum_rate_error_ppm:
            reasons.append("rate_error")
        if self._residual_p95_ns > self.config.maximum_residual_p95_ns:
            reasons.append("residual")
        if not self._mapping_causal:
            reasons.append("mapping_noncausal")
        if not 0 <= self._current_transport_delay_ns <= self.config.maximum_transport_delay_ns:
            reasons.append("transport_delay")
        return tuple(reasons)

    def reset(self) -> None:
        self._source_epoch += 1
        self._pairs.clear()
        self._last_source_ns = None
        self._last_receipt_ns = None
        self._slope = 1.0
        self._offset_ns = 0.0
        self._residual_p95_ns = math.inf
        self._source_origin_ns = 0
        self._receipt_origin_ns = 0
        self._span_ns = 0
        self._mapping_causal = False
        self._last_mapped_observation_ns = None
        self._observations_since_fit = 0
        self._current_transport_delay_ns = math.inf

    def observe(self, source_time_ns: int, receipt_time_ns: int) -> ClockMapEstimate:
        if not isinstance(source_time_ns, int) or source_time_ns <= 0:
            raise ClockSynchronizationError("source timestamp must be a positive integer")
        if not isinstance(receipt_time_ns, int) or receipt_time_ns <= 0:
            raise ClockSynchronizationError("receipt timestamp must be a positive integer")
        if self._last_receipt_ns is not None and receipt_time_ns <= self._last_receipt_ns:
            raise ClockSynchronizationError("receipt timestamp did not increase")
        if self._last_source_ns is not None and source_time_ns <= self._last_source_ns:
            self.reset()

        self._pairs.append((source_time_ns, receipt_time_ns))
        self._last_source_ns = source_time_ns
        self._last_receipt_ns = receipt_time_ns
        self._span_ns = self._pairs[-1][0] - self._pairs[0][0]
        self._observations_since_fit += 1
        if (
            len(self._pairs) <= self.config.minimum_samples
            or self._observations_since_fit >= self.config.refit_period_samples
        ):
            self._fit()
            self._observations_since_fit = 0
        mapped_time_ns = int(
            round(
                self._receipt_origin_ns
                + self._slope * (source_time_ns - self._source_origin_ns)
            )
        )
        self._mapping_causal = mapped_time_ns <= receipt_time_ns
        self._current_transport_delay_ns = receipt_time_ns - mapped_time_ns
        estimate = self.estimate(source_time_ns)
        if (
            self._last_mapped_observation_ns is not None
            and estimate.mapped_time_ns <= self._last_mapped_observation_ns
        ):
            self.reset()
            self._pairs.append((source_time_ns, receipt_time_ns))
            self._last_source_ns = source_time_ns
            self._last_receipt_ns = receipt_time_ns
            self._fit()
            self._observations_since_fit = 0
            self._mapping_causal = True
            self._current_transport_delay_ns = 0
            estimate = self.estimate(source_time_ns)
        self._last_mapped_observation_ns = estimate.mapped_time_ns
        return estimate

    def estimate(self, source_time_ns: int) -> ClockMapEstimate:
        if not isinstance(source_time_ns, int) or source_time_ns <= 0:
            raise ClockSynchronizationError("source timestamp must be a positive integer")
        if not self._pairs:
            raise ClockSynchronizationError("clock mapper has no observations")
        mapped = int(
            round(
                self._receipt_origin_ns
                + self._slope * (source_time_ns - self._source_origin_ns)
            )
        )
        return ClockMapEstimate(
            mapped_time_ns=mapped,
            source_epoch=self._source_epoch,
            valid=self.valid,
            fit_valid=self.fit_valid,
            sample_valid=self.sample_valid,
            health_reasons=self.health_reasons,
            sample_count=len(self._pairs),
            slope=self._slope,
            residual_p95_ns=self._residual_p95_ns,
            transport_delay_ns=int(self._current_transport_delay_ns),
        )

    def _fit(self) -> None:
        if len(self._pairs) == 1:
            source, receipt = self._pairs[0]
            self._source_origin_ns = source
            self._receipt_origin_ns = receipt
            self._slope = 1.0
            self._offset_ns = float(receipt - source)
            self._residual_p95_ns = math.inf
            self._span_ns = 0
            self._mapping_causal = True
            self._current_transport_delay_ns = 0
            return
        source = np.asarray([pair[0] for pair in self._pairs], dtype=np.int64)
        receipt = np.asarray([pair[1] for pair in self._pairs], dtype=np.int64)
        fit = fit_affine_lower_envelope_clock(
            source,
            receipt,
            lower_quantile=self.config.lower_envelope_quantile,
            rate_fit_quantile=self.config.rate_fit_quantile,
        )
        self._source_origin_ns = fit.source_origin_ns
        self._receipt_origin_ns = int(
            round(fit.receipt_origin_ns + fit.lower_envelope_residual_ns)
        )
        self._slope = fit.slope
        self._offset_ns = float(
            self._receipt_origin_ns - self._slope * self._source_origin_ns
        )
        self._residual_p95_ns = fit.residual_p95_ns
        self._span_ns = int(source[-1] - source[0])
        mapped_latest = int(
            round(
                self._receipt_origin_ns
                + self._slope * (int(source[-1]) - self._source_origin_ns)
            )
        )
        self._mapping_causal = mapped_latest <= int(receipt[-1])
