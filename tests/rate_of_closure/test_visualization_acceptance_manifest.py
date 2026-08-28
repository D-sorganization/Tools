"""Fail-closed R14.6 acceptance and human-review authority."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import pytest

from rate_of_closure.visualization_acceptance_manifest import (
    AcceptanceManifestError,
    load_visualization_acceptance_manifest,
)
from rate_of_closure.visualization_tab_manifest import (
    load_visualization_tab_manifest,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _document() -> dict[str, object]:
    path = (
        Path(__file__).parents[2]
        / "src"
        / "rate_of_closure"
        / "visualization_acceptance.v1.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _load(value: object) -> object:
    text = json.dumps(value)
    resource = type("Resource", (), {"read_text": lambda self, **_kwargs: text})()
    with patch("rate_of_closure.visualization_acceptance_manifest.files") as package:
        package.return_value.joinpath.return_value = resource
        return load_visualization_acceptance_manifest()


def test_acceptance_authority_covers_every_tab_state_and_reference_case() -> None:
    acceptance = load_visualization_acceptance_manifest()
    visibility = load_visualization_tab_manifest()

    assert acceptance.schema_id == "rate-of-closure/visualization-acceptance"
    assert acceptance.schema_version == 1
    assert acceptance.evidence_policy == (
        "registered-contract-not-rendered-or-human-approval"
    )
    assert acceptance.reference_cases == {
        "react": (
            "desktop-1440x900",
            "desktop-1280x720",
            "narrow-390x844",
        ),
        "pyqt": ("desktop-1440x900-dpi-1.0", "desktop-1440x900-dpi-1.5"),
    }
    assert tuple((entry.surface, entry.tab_id) for entry in acceptance.tabs) == tuple(
        (entry.surface, entry.tab_id) for entry in visibility.tabs
    )
    for entry, source in zip(acceptance.tabs, visibility.tabs, strict=True):
        assert tuple(entry.states) == tuple(source.states)
        assert all(
            state.descriptor == source.states[state_name]
            for state_name, state in entry.states.items()
        )
        assert tuple(entry.reference_cases) == acceptance.reference_cases[entry.surface]
        assert entry.context.frame
        assert entry.context.units
        assert entry.context.provenance
        assert entry.context.limitations
        assert entry.context.keyboard_path
        assert entry.context.nonvisual_alternative


def test_human_actions_remain_pending_and_cannot_be_promoted_by_automation() -> None:
    manifest = load_visualization_acceptance_manifest()
    root = Path(__file__).parents[2]

    assert tuple(manifest.human_actions) == (
        "manual-assistive-technology-protocol",
        "user-rendered-review-approval",
    )
    assert all(
        action.status == "pending-human" for action in manifest.human_actions.values()
    )
    assert all(
        action.evidence_identity is None for action in manifest.human_actions.values()
    )
    assert all(
        (root / action.protocol_path).is_file()
        for action in manifest.human_actions.values()
    )

    action = manifest.human_actions["manual-assistive-technology-protocol"]
    with pytest.raises(AcceptanceManifestError, match="human evidence"):
        replace(
            manifest,
            human_actions=MappingProxyType(
                {
                    **manifest.human_actions,
                    action.action_id: replace(
                        action,
                        status="approved",
                        evidence_identity="automated-test-run",
                    ),
                }
            ),
        ).validate()


def test_reader_rejects_coverage_context_and_state_drift() -> None:
    missing = _document()
    tabs = missing["tabs"]
    assert isinstance(tabs, list)
    tabs.pop()
    with pytest.raises(AcceptanceManifestError, match="visibility authority"):
        _load(missing)

    missing_context = _document()
    tabs = missing_context["tabs"]
    assert isinstance(tabs, list)
    first = tabs[0]
    assert isinstance(first, dict)
    context = first["context"]
    assert isinstance(context, dict)
    context.pop("units")
    with pytest.raises(AcceptanceManifestError, match="context fields"):
        _load(missing_context)

    false_reference = _document()
    references = false_reference["reference_cases"]
    assert isinstance(references, dict)
    react = references["react"]
    assert isinstance(react, list)
    react.pop()
    with pytest.raises(AcceptanceManifestError, match="reference cases"):
        _load(false_reference)


def test_acceptance_manifest_is_deeply_immutable() -> None:
    manifest = load_visualization_acceptance_manifest()

    assert isinstance(manifest.reference_cases, MappingProxyType)
    assert isinstance(manifest.tabs[0].states, MappingProxyType)
    assert isinstance(manifest.human_actions, MappingProxyType)
    with pytest.raises(TypeError):
        manifest.tabs[0].states["result"] = manifest.tabs[0].states["empty"]  # type: ignore[index]
