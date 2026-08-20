"""Strict noncoercive variation-plan identity wire contracts."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from shared.python.contracts import ContractViolationError
from shared.python.swing_sim.variation import VariationPlan

_FIXTURE = (
    Path(__file__).parents[5]
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "localized_torque_authoring_v1.json"
)


def _fixture() -> dict[str, Any]:
    return copy.deepcopy(json.loads(_FIXTURE.read_text(encoding="utf-8")))


def _noise(name: str, value: object) -> Callable[[dict[str, Any]], None]:
    def mutate(plan: dict[str, Any]) -> None:
        plan["noise"][0][name] = value

    return mutate


def _group(name: str, value: object) -> Callable[[dict[str, Any]], None]:
    def mutate(plan: dict[str, Any]) -> None:
        plan["groups"][0][name] = value

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_noise("spec_id", 7), "spec_id"),
        (_noise("spec_id", ""), "spec_id"),
        (_noise("spec_id", "bad\x00id"), "spec_id"),
        (_noise("point_ids", "joint.shoulder"), "point_ids"),
        (_noise("point_ids", [7]), "point_ids"),
        (_noise("point_ids", ["joint.\x7fshoulder"]), "point_ids"),
        (_noise("point_ids", ["joint.shoulder", "joint.shoulder"]), "point_ids"),
        (_noise("variable_key", 7), "variable_key"),
        (_noise("distribution", 7), "distribution"),
        (lambda plan: plan.__setitem__("mode", 7), "mode"),
        (lambda plan: plan.__setitem__("flight_model", 7), "flight_model"),
        (_group("group_id", 7), "group_id"),
        (_group("group_id", "bad\x1fid"), "group_id"),
        (_group("spec_ids", "shoulder-window"), "spec_ids"),
        (_group("spec_ids", [7, "wrist-window"]), "spec_ids"),
        (_group("spec_ids", ["shoulder-window", "shoulder-window"]), "spec_ids"),
        (_group("matrix_kind", 7), "matrix_kind"),
    ],
)
def test_plan_reader_rejects_coercive_identity_domains(
    mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    plan = _fixture()
    mutate(plan)

    with pytest.raises(ContractViolationError, match=message):
        VariationPlan.from_json_dict(plan)
