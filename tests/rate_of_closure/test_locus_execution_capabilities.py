"""Fail-closed contracts for whole-run and localized variation execution."""

from __future__ import annotations

import copy
import json
from importlib.resources import files

import pytest

from rate_of_closure.variation.locus_execution_capabilities import (
    LocusContractError,
    capability_for,
    load_locus_execution_contract,
    managed_registry_keys,
    parse_locus_execution_contract,
)
from shared.python.swing_sim.variation import variable_registry

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_contract_exhaustively_classifies_the_shared_registry() -> None:
    contract = load_locus_execution_contract()

    assert contract.schema_version == "rate-locus-execution-capabilities/v1"
    assert contract.mode == "multi_adapter"
    assert contract.source_kind == "declared_by_adapter"
    assert contract.point_id_semantics == (
        "topological_control_loci_not_spatial_trace_points"
    )
    assert set(contract.capabilities) == managed_registry_keys(variable_registry())

    supported = {
        key for key, capability in contract.capabilities.items() if capability.supported
    }
    assert len(supported) == 19
    assert all(
        capability.unsupported_reason is not None
        for capability in contract.capabilities.values()
        if not capability.supported
    )


def test_typed_contract_round_trips_the_packaged_authority_exactly() -> None:
    payload = json.loads(
        files("rate_of_closure")
        .joinpath("locus_execution_capabilities.v1.json")
        .read_text(encoding="utf-8")
    )

    assert load_locus_execution_contract().to_wire() == payload


def test_contract_distinguishes_whole_run_from_exact_joint_windows() -> None:
    global_capability = capability_for("swing_sim.swing.yaw_deg")
    shoulder = capability_for("swing_sim.swing.shoulder_commanded_torque_offset_nm")

    assert global_capability.adapter_id == "global_simulation_value/v1"
    assert global_capability.whole_run is True
    assert global_capability.time_window_policy == "forbidden"
    assert global_capability.point_locus_policy == "forbidden"
    assert global_capability.point_ids == ()

    assert shoulder.adapter_id == "localized_joint_torque_offset/v1"
    assert shoulder.whole_run is False
    assert shoulder.time_window_policy == "required_half_open_seconds"
    assert shoulder.point_locus_policy == "required_exact_topological"
    assert shoulder.point_ids == ("joint.shoulder",)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["capabilities"].pop(),
            "registry coverage",
        ),
        (
            lambda payload: payload["capabilities"].append(
                copy.deepcopy(payload["capabilities"][0])
            ),
            "duplicate variable_key",
        ),
        (
            lambda payload: payload["capabilities"][0].update(
                {"time_window_policy": "sometimes"}
            ),
            "time_window_policy",
        ),
        (
            lambda payload: payload["capabilities"][0].update(
                {"whole_run": True, "time_window_policy": "required_half_open_seconds"}
            ),
            "whole-run capability cannot require",
        ),
        (
            lambda payload: payload["capabilities"][0].update(
                {"supported": False, "unsupported_reason": None}
            ),
            "unsupported_reason",
        ),
    ],
)
def test_parser_rejects_malformed_or_registry_drifting_authority(
    mutation: object,
    message: str,
) -> None:
    payload = load_locus_execution_contract().to_wire()
    assert callable(mutation)
    mutation(payload)  # type: ignore[operator]

    with pytest.raises(LocusContractError, match=message):
        parse_locus_execution_contract(
            payload,
            registered_keys=managed_registry_keys(variable_registry()),
        )


def test_unknown_variable_lookup_fails_closed() -> None:
    with pytest.raises(LocusContractError, match="not declared"):
        capability_for("swing_sim.swing.future_undeclared_input")
