"""Force-only time integration must reuse work without computing curvature."""

from collections.abc import Callable

import numpy as np
import pytest

from shared.python.golf_club import _shaft_section as section_module
from shared.python.golf_club._shaft_moving_chain import moving_chain_response

from .test_shaft_moving_chain import _axial_case, _nonplanar_case


@pytest.mark.parametrize("factory", [_axial_case, _nonplanar_case])
def test_work_matches_full_tangent_without_evaluating_curvature(
    factory: Callable, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain, initial, controls = factory()
    elastic = chain.shaft.elastic
    expected = elastic.linearize(initial.poses)
    response = moving_chain_response(chain, initial, controls)

    def forbidden(*args: object) -> None:
        raise AssertionError("unused energy curvature was evaluated")

    monkeypatch.setattr(section_module, "_map_derivative", forbidden)
    actual = elastic.work(initial.poses)
    np.testing.assert_array_equal(actual.residual, expected.residual)
    assert actual.elastic_energy_j == expected.elastic_energy_j
    repeated = moving_chain_response(chain, initial, controls)
    np.testing.assert_array_equal(repeated.twist_rates, response.twist_rates)
    assert repeated.total_energy_j == response.total_energy_j
    assert repeated.energy_rate_w == response.energy_rate_w
