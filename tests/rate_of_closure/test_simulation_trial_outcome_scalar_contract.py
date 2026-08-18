"""Scalar-domain closure tests for complete Rate trial outcomes."""

from __future__ import annotations

import numpy as np
import pytest

from rate_of_closure.variation.ensemble_io import from_json_dict, to_json_dict
from rate_of_closure.variation.simulation_types import (
    ALL_OUTPUT_NAMES,
    EVALUATED_HIT,
    SimulationEnsembleResult,
    SimulationTrialOutcome,
)
from shared.python.contracts import ContractViolationError

from .test_variation_ensemble_io_reader import _result

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _hit_values() -> dict[str, object]:
    return {name: float(index + 1) for index, name in enumerate(ALL_OUTPUT_NAMES)}


@pytest.mark.parametrize("invalid", [True, np.bool_(True), "1.0"])
def test_outcome_rejects_noncanonical_available_scalars(invalid: object) -> None:
    values = _hit_values()
    values[ALL_OUTPUT_NAMES[0]] = invalid

    with pytest.raises(ContractViolationError, match="real numbers excluding booleans"):
        SimulationTrialOutcome(0, EVALUATED_HIT, values)


@pytest.mark.parametrize("scalar", [np.float32(1.0), np.int64(1)])
def test_numpy_real_scalar_normalizes_and_round_trips(scalar: object) -> None:
    source = _result()
    values = dict(source.outcomes[0].values)
    values[ALL_OUTPUT_NAMES[0]] = scalar
    outcome = SimulationTrialOutcome(0, EVALUATED_HIT, values)
    result = SimulationEnsembleResult(
        (outcome, *source.outcomes[1:]), source.variation, source.traces
    )

    assert type(outcome.value(ALL_OUTPUT_NAMES[0])) is float
    assert to_json_dict(from_json_dict(to_json_dict(result))) == to_json_dict(result)
