"""Strict numeric-domain and allocation tests for ensemble chunks."""

from __future__ import annotations

import numpy as np
import pytest

from rate_of_closure.simulation import ContactMode
from rate_of_closure.variation._ensemble_limits import MAX_INPUT_CELLS
from rate_of_closure.variation.ensemble_chunks import SimulationResultChunk
from rate_of_closure.variation.simulation_adapter import run_simulation_ensemble
from shared.python.contracts import ContractViolationError

from .test_variation_simulation_adapter import _config, _request

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("inputs", np.array([[True]], dtype=bool)),
        ("inputs", np.array([["1.25"]], dtype=str)),
        ("positions", np.array([[[[True, False, True]]]], dtype=bool)),
        ("positions", np.array([[[["1", "2", "3"]]]], dtype=str)),
    ],
)
def test_chunk_rejects_coercive_scientific_array_domains(
    field: str, invalid: np.ndarray
) -> None:
    source = run_simulation_ensemble(
        _request((_config(ContactMode.DELIVERY_INSPECTION),))
    )
    inputs = invalid if field == "inputs" else source.variation.inputs
    positions = (
        np.broadcast_to(invalid, source.traces.positions_m.shape)
        if field == "positions"
        else source.traces.positions_m
    )

    with pytest.raises(ContractViolationError, match="real non-boolean"):
        SimulationResultChunk(
            0,
            inputs,
            source.outcomes,
            positions,
            source.traces.sample_valid,
            source.traces.impact_sample_indices,
        )


def test_chunk_input_cells_are_bounded_before_owned_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = run_simulation_ensemble(
        _request((_config(ContactMode.DELIVERY_INSPECTION),))
    )
    oversized = np.lib.stride_tricks.as_strided(
        np.zeros(1), shape=(1, MAX_INPUT_CELLS + 1), strides=(0, 0)
    )

    def unexpected_copy(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("oversized input array copied before its cap")

    monkeypatch.setattr(
        "rate_of_closure.variation.ensemble_chunks._owned_array", unexpected_copy
    )
    with pytest.raises(ContractViolationError, match="input cell limit"):
        SimulationResultChunk(
            0,
            oversized,
            source.outcomes,
            source.traces.positions_m,
            source.traces.sample_valid,
            source.traces.impact_sample_indices,
        )


@pytest.mark.parametrize(
    "invalid",
    [
        np.array([50.9], dtype=float),
        np.array(["50"], dtype=str),
        np.array([True], dtype=bool),
    ],
)
def test_chunk_rejects_non_integer_impact_domains(invalid: np.ndarray) -> None:
    source = run_simulation_ensemble(
        _request((_config(ContactMode.DELIVERY_INSPECTION),))
    )

    with pytest.raises(ContractViolationError, match="genuine integer"):
        SimulationResultChunk(
            0,
            source.variation.inputs,
            source.outcomes,
            source.traces.positions_m,
            source.traces.sample_valid,
            invalid,
        )


def test_chunk_rejects_unsigned_impact_overflow_before_conversion() -> None:
    source = run_simulation_ensemble(
        _request((_config(ContactMode.FIXED_BALL_CONTACT),))
    )

    with pytest.raises(ContractViolationError, match="integer range"):
        SimulationResultChunk(
            0,
            source.variation.inputs,
            source.outcomes,
            source.traces.positions_m,
            source.traces.sample_valid,
            np.array([np.iinfo(np.uint64).max], dtype=np.uint64),
        )
