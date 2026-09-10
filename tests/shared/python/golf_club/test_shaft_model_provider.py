"""Versioned coefficients compose into the existing moving-chain mechanics."""

from dataclasses import replace

import numpy as np
import pytest

from shared.python.golf_club._shaft_model_provider import compile_shaft_model
from shared.python.golf_club._shaft_moving_chain import moving_chain_response
from shared.python.golf_club.shaft_model_data import (
    DistributedShaftModel,
    ShaftModelSection,
    shaft_model_from_json,
    shaft_model_to_json,
)

from .test_shaft_model_data import _load, _payload
from .test_shaft_trajectory_rotation import _problem


def test_loaded_coefficients_preserve_full_moving_chain_response() -> None:
    problem, initial = _problem(2)
    chain = problem.chain
    shaft = chain.shaft
    source = _load(_payload()).sources[0]
    records = tuple(
        ShaftModelSection(section, inertia, "fixture", "fixture")
        for section, inertia in zip(shaft.sections, shaft.inertias, strict=True)
    )
    # These are exactly the pre-existing synthetic fixture's supplied parameters.
    source = replace(source, method="existing rotating trajectory fixture")
    material_frame = next(iter(records[0].material_frame_ids))
    model = DistributedShaftModel(
        "rotating-fixture", material_frame, records, (source,)
    )
    loaded = shaft_model_from_json(shaft_model_to_json(model))
    compiled = compile_shaft_model(loaded, shaft.frame, shaft.elastic.loads)
    result = moving_chain_response(
        replace(chain, shaft=compiled), initial, problem.controls
    )
    expected = moving_chain_response(chain, initial, problem.controls)
    np.testing.assert_array_equal(result.twist_rates, expected.twist_rates)
    assert result.shaft_kinetic_energy_j == expected.shaft_kinetic_energy_j
    assert result.elastic_energy_j == expected.elastic_energy_j
    assert result.anchor_power_w == expected.anchor_power_w


def test_compilation_keeps_explicit_frame_and_existing_load_validation() -> None:
    problem, _ = _problem(2)
    model = _load(_payload())
    frame = problem.chain.shaft.frame
    with pytest.raises(TypeError):
        compile_shaft_model(model, object())
    with pytest.raises(TypeError):
        compile_shaft_model(model, frame, (object(),))
    with pytest.raises(TypeError):
        compile_shaft_model(object(), frame)
