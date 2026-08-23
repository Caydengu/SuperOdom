import numpy as np
import pytest

from g1_root_state_bridge.clock_sync import (
    ClockMapConfig,
    ClockSampleDisposition,
    ClockSynchronizationError,
    OnlineAffineClockMapper,
    classify_clock_estimate,
    fit_affine_lower_envelope_clock,
)


def test_affine_clock_becomes_valid_and_maps_source_time() -> None:
    mapper = OnlineAffineClockMapper(
        ClockMapConfig(
            minimum_samples=4,
            maximum_samples=16,
            minimum_span_ns=1,
            maximum_residual_p95_ns=1_000,
            refit_period_samples=1,
        )
    )
    offset = 123_000_000
    estimates = [
        mapper.observe(1_000_000_000 + index * 10_000_000, 1_000_000_000 + index * 10_000_000 + offset)
        for index in range(6)
    ]
    assert estimates[-1].valid
    assert estimates[-1].fit_valid
    assert estimates[-1].sample_valid
    assert estimates[-1].health_reasons == ()
    assert estimates[-1].mapped_time_ns == 1_050_000_000 + offset
    assert estimates[-1].residual_p95_ns < 1.0


def test_source_regression_starts_new_invalid_epoch() -> None:
    mapper = OnlineAffineClockMapper(
        ClockMapConfig(
            minimum_samples=3,
            minimum_span_ns=1,
            refit_period_samples=1,
        )
    )
    for index in range(3):
        assert mapper.observe(10_000 + index * 100, 20_000 + index * 100).source_epoch == 0
    assert mapper.valid
    reset = mapper.observe(5_000, 21_000)
    assert reset.source_epoch == 1
    assert not reset.valid
    assert reset.sample_count == 1


def test_receipt_regression_is_rejected() -> None:
    mapper = OnlineAffineClockMapper()
    mapper.observe(100, 200)
    with pytest.raises(ClockSynchronizationError, match="receipt timestamp"):
        mapper.observe(101, 199)


def test_excessive_rate_error_never_becomes_valid() -> None:
    mapper = OnlineAffineClockMapper(
        ClockMapConfig(
            minimum_samples=4,
            minimum_span_ns=1,
            maximum_rate_error_ppm=100.0,
            refit_period_samples=1,
        )
    )
    for index in range(5):
        estimate = mapper.observe(1_000_000 + index * 1_000, 2_000_000 + index * 2_000)
    assert not estimate.valid


def test_delayed_transport_does_not_poison_rate_fit_but_rejects_sample() -> None:
    mapper = OnlineAffineClockMapper(
        ClockMapConfig(
            minimum_samples=6,
            maximum_samples=16,
            minimum_span_ns=1,
            maximum_residual_p95_ns=1_000_000,
            maximum_rate_error_ppm=100_000.0,
            refit_period_samples=1,
        )
    )
    for index in range(12):
        jitter_ns = 20_000_000 if index % 2 else 0
        estimate = mapper.observe(
            1_000_000_000 + index * 30_000_000,
            2_000_000_000 + index * 30_000_000 + jitter_ns,
        )
    assert estimate.residual_p95_ns < 1_000_000
    assert estimate.transport_delay_ns > 10_000_000
    assert estimate.fit_valid
    assert not estimate.sample_valid
    assert estimate.health_reasons == ("transport_delay",)
    assert not estimate.valid


def test_batch_lower_envelope_maps_to_least_delayed_clock_domain() -> None:
    source = np.asarray([1_000, 2_000, 3_000, 4_000, 5_000], dtype=np.int64)
    receipt = source + np.asarray([100, 110, 150, 120, 130], dtype=np.int64)
    fit = fit_affine_lower_envelope_clock(source, receipt, lower_quantile=0.0)
    mapped = np.asarray([fit.map_time_ns(int(value)) for value in source])
    assert np.all(mapped <= receipt)
    assert abs(fit.slope - 1.0) < 0.02


def test_current_transport_delay_is_rejected_without_poisoning_clock_fit() -> None:
    mapper = OnlineAffineClockMapper(
        ClockMapConfig(
            minimum_samples=4,
            minimum_span_ns=1,
            maximum_transport_delay_ns=5_000_000,
            refit_period_samples=32,
        )
    )
    for index in range(4):
        estimate = mapper.observe(
            1_000_000_000 + index * 10_000_000,
            2_000_000_000 + index * 10_000_000,
        )
    assert estimate.valid
    delayed = mapper.observe(1_040_000_000, 2_060_000_000)
    assert delayed.transport_delay_ns > 5_000_000
    assert delayed.fit_valid
    assert not delayed.sample_valid
    assert delayed.health_reasons == ("transport_delay",)
    assert not delayed.valid
    assert (
        classify_clock_estimate(delayed, lane_active=True)
        is ClockSampleDisposition.DROP_SAMPLE
    )


def test_unhealthy_fit_resets_only_an_active_lane() -> None:
    mapper = OnlineAffineClockMapper()
    estimate = mapper.observe(1_000_000_000, 2_000_000_000)
    assert not estimate.fit_valid
    assert (
        classify_clock_estimate(estimate, lane_active=False)
        is ClockSampleDisposition.DROP_SAMPLE
    )
    assert (
        classify_clock_estimate(estimate, lane_active=True)
        is ClockSampleDisposition.RESET_LANE
    )
