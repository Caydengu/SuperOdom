from __future__ import annotations

import copy

import pytest

from scripts.fit_motive_map_transform import fit_control_points
from scripts.validate_motive_polycam_transform import content_sha256, validate_transform
from tests.root_state.test_motive_map_transform import _fixture


def _accepted_transform():
    value = fit_control_points(_fixture())
    value.pop("content_sha256")
    value["pelvis_heading"] = {
        "rigid_body_id": 42,
        "rigid_body_name": "G1_PELVIS_F_4123",
        "forward_axis": "+x",
        "yaw_offset_rad": 0.31,
    }
    value["evaluator_ready"] = True
    value["content_sha256"] = content_sha256(value)
    return value


def _validate(value):
    return validate_transform(
        value,
        expected_glb_sha256="a" * 64,
        expected_surface_sha256="b" * 64,
        expected_structural_map_sha256="c" * 64,
        expected_pelvis_rigid_body_id=42,
        expected_pelvis_rigid_body_name="G1_PELVIS_F_4123",
    )


def test_transform_validator_accepts_exact_map_and_pelvis_contract():
    result = _validate(_accepted_transform())
    assert result["status"] == "pass"
    assert result["pelvis_rigid_body_id"] == 42


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["map_identity"].update(glb_sha256="d" * 64), "different Polycam"),
        (lambda value: value["pelvis_heading"].update(rigid_body_id=39), "rigid-body ID"),
        (lambda value: value["T_map_from_motive"][2].__setitem__(1, -1.0), "not proper"),
    ],
)
def test_transform_validator_fails_closed_on_identity_and_axis_errors(mutation, message):
    value = copy.deepcopy(_accepted_transform())
    mutation(value)
    value["content_sha256"] = content_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match=message):
        _validate(value)


def test_transform_validator_rejects_tampering_even_when_fields_look_valid():
    value = _accepted_transform()
    value["pelvis_heading"]["yaw_offset_rad"] = 0.5
    with pytest.raises(ValueError, match="content digest"):
        _validate(value)
