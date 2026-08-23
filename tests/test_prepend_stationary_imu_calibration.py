import pytest

from scripts.prepend_stationary_imu_calibration import shifted_time_ns


def test_shifted_time_preserves_relative_cadence() -> None:
    assert shifted_time_ns(
        1_700_000_000_015,
        source_origin_ns=1_700_000_000_000,
        target_origin_ns=1_800_000_000_000,
    ) == 1_800_000_000_015


def test_shifted_time_accepts_only_integer_contract() -> None:
    with pytest.raises(ValueError):
        # int() is deliberately used for rosbag stamps; a nonsensical value
        # must still fail rather than silently becoming a valid timestamp.
        shifted_time_ns(
            float("nan"),
            source_origin_ns=1,
            target_origin_ns=2,
        )
