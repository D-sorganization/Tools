"""Strict R14.6 cross-surface acceptance and human-review authority."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from types import MappingProxyType
from typing import Any

from rate_of_closure.visualization_tab_manifest import (
    load_visualization_tab_manifest,
)

_REFERENCE_CASES = {
    "react": (
        "desktop-1440x900",
        "desktop-1280x720",
        "narrow-390x844",
    ),
    "pyqt": ("desktop-1440x900-dpi-1.0", "desktop-1440x900-dpi-1.5"),
}
_CONTEXT_FIELDS = {
    "frame",
    "units",
    "provenance",
    "limitations",
    "keyboard_path",
    "nonvisual_alternative",
}
_HUMAN_ACTIONS = {
    "manual-assistive-technology-protocol",
    "user-rendered-review-approval",
}


class AcceptanceManifestError(ValueError):
    """Raised when the R14.6 acceptance authority is incomplete or overstated."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AcceptanceManifestError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise AcceptanceManifestError(f"non-finite JSON value: {value}")


def _object(value: object, keys: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise AcceptanceManifestError(f"{context} fields must be exact")
    return value


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 300:
        raise AcceptanceManifestError(f"{context} must be bounded nonempty text")
    return value


@dataclass(frozen=True)
class AcceptanceState:
    """One visibility-authority lifecycle state and its applicability."""

    descriptor: str
    applicable: bool


@dataclass(frozen=True)
class ScientificContext:
    """Scientific interpretation and nonvisual access bound to one tab."""

    frame: str
    units: str
    provenance: str
    limitations: str
    keyboard_path: str
    nonvisual_alternative: str


@dataclass(frozen=True)
class AcceptanceTab:
    """One tab expanded over every registered state and reference case."""

    surface: str
    tab_id: str
    reference_cases: tuple[str, ...]
    states: Mapping[str, AcceptanceState]
    context: ScientificContext


@dataclass(frozen=True)
class HumanAction:
    """Evidence that automation is forbidden to claim on a human's behalf."""

    action_id: str
    status: str
    protocol_path: str
    evidence_identity: str | None


@dataclass(frozen=True)
class VisualizationAcceptanceManifest:
    """Immutable R14.6 acceptance coverage and retained human boundary."""

    schema_id: str
    schema_version: int
    evidence_policy: str
    reference_cases: Mapping[str, tuple[str, ...]]
    tabs: tuple[AcceptanceTab, ...]
    human_actions: Mapping[str, HumanAction]

    def validate(self) -> None:
        """Reject coverage drift, false state claims, and automated human approval."""
        if dict(self.reference_cases) != _REFERENCE_CASES:
            raise AcceptanceManifestError("reference cases must match v1 authority")
        visibility = load_visualization_tab_manifest()
        identities = tuple((entry.surface, entry.tab_id) for entry in self.tabs)
        expected = tuple((entry.surface, entry.tab_id) for entry in visibility.tabs)
        if identities != expected or len(set(identities)) != len(identities):
            raise AcceptanceManifestError(
                "acceptance tabs must exactly match visibility authority"
            )
        for entry, source in zip(self.tabs, visibility.tabs, strict=True):
            if entry.reference_cases != _REFERENCE_CASES[entry.surface]:
                raise AcceptanceManifestError(
                    "tab reference-case coverage is incomplete"
                )
            if tuple(entry.states) != tuple(source.states):
                raise AcceptanceManifestError(
                    "tab states must match visibility authority"
                )
            for state_name, state in entry.states.items():
                if state.descriptor != source.states[state_name]:
                    raise AcceptanceManifestError("state descriptor drift")
                expected_applicability = state.descriptor != "not-applicable"
                if state.applicable is not expected_applicability:
                    raise AcceptanceManifestError("state applicability drift")
        if set(self.human_actions) != _HUMAN_ACTIONS:
            raise AcceptanceManifestError("human action coverage is incomplete")
        for action in self.human_actions.values():
            if action.status != "pending-human" or action.evidence_identity is not None:
                raise AcceptanceManifestError(
                    "human evidence cannot be supplied by the v1 automation authority"
                )


def _parse_context(value: object) -> ScientificContext:
    context = _object(value, _CONTEXT_FIELDS, "context")
    return ScientificContext(
        frame=_text(context["frame"], "frame"),
        units=_text(context["units"], "units"),
        provenance=_text(context["provenance"], "provenance"),
        limitations=_text(context["limitations"], "limitations"),
        keyboard_path=_text(context["keyboard_path"], "keyboard path"),
        nonvisual_alternative=_text(
            context["nonvisual_alternative"], "nonvisual alternative"
        ),
    )


def _parse_human_action(value: object) -> HumanAction:
    action = _object(
        value,
        {"action_id", "status", "protocol_path", "evidence_identity"},
        "human action",
    )
    identity = action["evidence_identity"]
    if identity is not None:
        identity = _text(identity, "human evidence identity")
    return HumanAction(
        action_id=_text(action["action_id"], "human action id"),
        status=_text(action["status"], "human action status"),
        protocol_path=_text(action["protocol_path"], "human protocol path"),
        evidence_identity=identity,
    )


def load_visualization_acceptance_manifest() -> VisualizationAcceptanceManifest:
    """Load and expand the packaged acceptance authority over states and cases."""
    resource = files("rate_of_closure").joinpath("visualization_acceptance.v1.json")
    try:
        raw = json.loads(
            resource.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AcceptanceManifestError("acceptance manifest is unreadable") from exc
    document = _object(
        raw,
        {
            "schema_id",
            "schema_version",
            "evidence_policy",
            "coverage_policy",
            "reference_cases",
            "human_actions",
            "tabs",
        },
        "manifest",
    )
    version = document["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise AcceptanceManifestError("schema version must be an integer")
    references = document["reference_cases"]
    tabs = document["tabs"]
    actions = document["human_actions"]
    if not isinstance(references, dict) or not isinstance(tabs, list):
        raise AcceptanceManifestError("reference cases and tabs must be containers")
    if not isinstance(actions, list):
        raise AcceptanceManifestError("human actions must be an array")
    parsed_references = MappingProxyType(
        {
            _text(surface, "reference surface"): tuple(
                _text(case, "reference case") for case in cases
            )
            for surface, cases in references.items()
            if isinstance(cases, list)
        }
    )
    visibility = load_visualization_tab_manifest()
    source_by_identity = {
        (entry.surface, entry.tab_id): entry for entry in visibility.tabs
    }
    parsed_tabs: list[AcceptanceTab] = []
    for value in tabs:
        tab = _object(value, {"surface", "tab_id", "context"}, "tab")
        surface = _text(tab["surface"], "surface")
        tab_id = _text(tab["tab_id"], "tab id")
        source = source_by_identity.get((surface, tab_id))
        if source is None or surface not in parsed_references:
            raise AcceptanceManifestError("unknown acceptance tab identity")
        states = MappingProxyType(
            {
                name: AcceptanceState(
                    descriptor=descriptor,
                    applicable=descriptor != "not-applicable",
                )
                for name, descriptor in source.states.items()
            }
        )
        parsed_tabs.append(
            AcceptanceTab(
                surface=surface,
                tab_id=tab_id,
                reference_cases=parsed_references[surface],
                states=states,
                context=_parse_context(tab["context"]),
            )
        )
    parsed_actions = tuple(_parse_human_action(value) for value in actions)
    action_map = MappingProxyType(
        {action.action_id: action for action in parsed_actions}
    )
    if len(action_map) != len(parsed_actions):
        raise AcceptanceManifestError("duplicate human action identity")
    manifest = VisualizationAcceptanceManifest(
        schema_id=_text(document["schema_id"], "schema id"),
        schema_version=version,
        evidence_policy=_text(document["evidence_policy"], "evidence policy"),
        reference_cases=parsed_references,
        tabs=tuple(parsed_tabs),
        human_actions=action_map,
    )
    if (
        manifest.schema_id != "rate-of-closure/visualization-acceptance"
        or manifest.schema_version != 1
        or manifest.evidence_policy
        != "registered-contract-not-rendered-or-human-approval"
        or document["coverage_policy"]
        != "all-visibility-states-x-surface-reference-cases"
    ):
        raise AcceptanceManifestError("unsupported acceptance manifest")
    manifest.validate()
    return manifest


__all__ = [
    "AcceptanceManifestError",
    "AcceptanceState",
    "AcceptanceTab",
    "HumanAction",
    "ScientificContext",
    "VisualizationAcceptanceManifest",
    "load_visualization_acceptance_manifest",
]
