"""Cross-runtime contracts for workspace torque-profile selection."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from rate_of_closure.application.workspace_torque_session import (
    LegacyTorqueMigrationRequired,
    TorqueWorkspaceState,
    migrate_legacy_torque_fallback,
    torque_workspace_from_payload,
    torque_workspace_to_payload,
)
from shared.python.swing_sim.run_config import DoublePendulumRunConfig
from shared.python.swing_sim.torque_profiles import PrescribedTorqueProfile

_FIXTURE = (
    Path(__file__).parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__/workspace_torque_parity.json"
)


def _raw_fixture() -> dict[str, object]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _profiles(raw: dict[str, object]) -> tuple[PrescribedTorqueProfile, ...]:
    values = raw["profiles"]
    assert isinstance(values, list)
    return tuple(PrescribedTorqueProfile.from_json_dict(item) for item in values)


def test_cross_runtime_fixture_round_trips_without_coefficient_or_fit_loss() -> None:
    raw = _raw_fixture()
    profiles = _profiles(raw)

    state = torque_workspace_from_payload(raw["selection"], profiles)

    assert state == TorqueWorkspaceState(
        profiles=profiles,
        active_profile_id="profile.web_parity.v1",
        run_config=DoublePendulumRunConfig.prescribed(
            "profile.web_parity.v1",
            joint_locks=state.run_config.joint_locks,
        ),
    )
    assert torque_workspace_to_payload(state) == raw["selection"]
    assert profiles[0].to_json_dict() == raw["profiles"][0]


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("schema_version",), 2, "unsupported"),
        (("data", "run_config", "locked_joint_ids"), ["joint.unknown"], "joint"),
        (("data", "selection_provenance", "profile_source"), "drawn", "source"),
    ],
)
def test_selection_rejects_schema_joint_and_provenance_corruption(
    path: tuple[str, ...], value: object, match: str
) -> None:
    raw = _raw_fixture()
    selection = deepcopy(raw["selection"])
    target = selection
    for key in path[:-1]:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    target[path[-1]] = value

    with pytest.raises((TypeError, ValueError), match=match):
        torque_workspace_from_payload(selection, _profiles(raw))


def test_legacy_migration_is_explicit_and_rejects_a_conflicting_library() -> None:
    raw = _raw_fixture()
    state = torque_workspace_from_payload(raw["selection"], _profiles(raw))

    assert migrate_legacy_torque_fallback(state, ()) == state

    conflicting = PrescribedTorqueProfile.from_json_dict(
        {**raw["profiles"][0], "profile_id": "profile.conflict.v1"}
    )
    with pytest.raises(LegacyTorqueMigrationRequired, match="conflicts"):
        migrate_legacy_torque_fallback(state, (conflicting,))


def test_legacy_migration_compares_library_identity_independent_of_order() -> None:
    raw = _raw_fixture()
    first = PrescribedTorqueProfile.from_json_dict(
        {**raw["profiles"][0], "profile_id": "profile.a.v1"}
    )
    second = PrescribedTorqueProfile.from_json_dict(
        {**raw["profiles"][0], "profile_id": "profile.b.v1"}
    )
    fallback = TorqueWorkspaceState(
        profiles=(second, first),
        active_profile_id=second.profile_id,
        run_config=DoublePendulumRunConfig(),
    )

    assert migrate_legacy_torque_fallback(fallback, (second, first)) == fallback
