from __future__ import annotations

from dataclasses import replace

import pytest

from g1_root_state_bridge.joint_buffer import JointBuffer, JointSynchronizationError
from g1_root_state_bridge.joint_contract import CANONICAL_G1_JOINT_NAMES, TimedJointSample


def sample(stamp_ns: int, sequence: int, offset: float) -> TimedJointSample:
    return TimedJointSample(
        stamp_ns=stamp_ns,
        receipt_ns=stamp_ns + 2_000_000,
        names=CANONICAL_G1_JOINT_NAMES,
        position=tuple(offset + index for index in range(29)),
        velocity=tuple(2.0 * offset + index for index in range(29)),
        sequence=sequence,
        source_epoch=7,
    )


def test_buffer_interpolates_position_and_velocity_inside_bracket() -> None:
    buffer = JointBuffer(capacity=8)
    buffer.append(sample(1_000_000_000, 10, 0.0))
    buffer.append(sample(1_010_000_000, 11, 10.0))

    result = buffer.interpolate(1_005_000_000, max_gap_ns=10_000_000)

    assert result.stamp_ns == 1_005_000_000
    assert result.receipt_ns == 1_012_000_000
    assert result.sequence == 11
    assert result.names == CANONICAL_G1_JOINT_NAMES
    assert result.position[0] == 5.0
    assert result.position[12:15] == (17.0, 18.0, 19.0)
    assert result.velocity[0] == 10.0


def test_buffer_canonicalizes_unordered_input_before_interpolation() -> None:
    first = sample(1_000_000_000, 10, 0.0)
    reverse_indices = tuple(reversed(range(29)))
    unordered = replace(
        first,
        names=tuple(first.names[index] for index in reverse_indices),
        position=tuple(first.position[index] for index in reverse_indices),
        velocity=tuple(first.velocity[index] for index in reverse_indices),
    )
    buffer = JointBuffer(capacity=8)
    buffer.append(unordered)
    buffer.append(sample(1_010_000_000, 11, 10.0))

    result = buffer.interpolate(1_005_000_000, max_gap_ns=10_000_000)
    assert result.position[12:15] == (17.0, 18.0, 19.0)


def test_buffer_returns_an_exact_timestamp_without_interpolation() -> None:
    exact = sample(1_000_000_000, 10, 3.0)
    buffer = JointBuffer(capacity=8)
    buffer.append(exact)
    assert buffer.interpolate(exact.stamp_ns, max_gap_ns=10_000_000) == exact


@pytest.mark.parametrize(
    ("query_ns", "max_gap_ns", "match"),
    [
        (999_000_000, 10_000_000, "outside"),
        (1_031_000_000, 10_000_000, "outside"),
        (1_015_000_000, 10_000_000, "gap"),
    ],
)
def test_buffer_rejects_unbracketed_or_excessive_gap(
    query_ns: int, max_gap_ns: int, match: str
) -> None:
    buffer = JointBuffer(capacity=8)
    buffer.append(sample(1_000_000_000, 10, 0.0))
    buffer.append(sample(1_030_000_000, 11, 10.0))
    with pytest.raises(JointSynchronizationError, match=match):
        buffer.interpolate(query_ns, max_gap_ns=max_gap_ns)


@pytest.mark.parametrize(
    ("second", "match"),
    [
        (sample(1_000_000_000, 11, 1.0), "timestamp"),
        (sample(1_001_000_000, 10, 1.0), "sequence"),
    ],
)
def test_buffer_rejects_nonmonotonic_samples(second: TimedJointSample, match: str) -> None:
    buffer = JointBuffer(capacity=8)
    buffer.append(sample(1_000_000_000, 10, 0.0))
    with pytest.raises(JointSynchronizationError, match=match):
        buffer.append(second)


def test_new_monotonic_source_epoch_resets_the_interpolation_window() -> None:
    buffer = JointBuffer(capacity=8)
    buffer.append(sample(1_000_000_000, 50, 0.0))
    restarted = replace(sample(1_001_000_000, 1, 1.0), source_epoch=8)
    buffer.append(restarted)

    assert buffer.interpolate(restarted.stamp_ns, max_gap_ns=10_000_000) == restarted
    with pytest.raises(JointSynchronizationError, match="outside"):
        buffer.interpolate(1_000_000_000, max_gap_ns=10_000_000)


def test_older_source_epoch_is_rejected() -> None:
    buffer = JointBuffer(capacity=8)
    buffer.append(sample(1_000_000_000, 50, 0.0))
    with pytest.raises(JointSynchronizationError, match="epoch"):
        buffer.append(replace(sample(1_001_000_000, 51, 1.0), source_epoch=6))
