"""Shared contracts for the proximal-distal article companion.

The catalog is deliberately UI-neutral.  PyQt and React consume the same JSON
resource so experiment language, limitations, and glossary definitions cannot
drift between interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json
from typing import Any, Mapping

_CATALOG_RESOURCE = "resources/companion_catalog.json"
_RUN_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class Experiment:
    """A bounded, falsifiable learning experiment."""

    id: str
    title: str
    model: str
    purpose: str
    hypothesis: str
    falsifier: str
    workflow: tuple[str, ...]
    tips: tuple[str, ...]
    observables: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class GlossaryTerm:
    """A scientific term and its reader-facing explanation."""

    id: str
    term: str
    definition: str
    plain_language: str
    units: str
    caution: str


@dataclass(frozen=True)
class CompanionCatalog:
    """Validated companion metadata shared by every interface."""

    schema_version: str
    title: str
    scientific_status: str
    experiments: tuple[Experiment, ...]
    glossary: tuple[GlossaryTerm, ...]


def _required_text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value.strip():
        raise ValueError(f"{field} must not be empty")
    return value


def _required_text_tuple(record: Mapping[str, Any], field: str) -> tuple[str, ...]:
    value = record.get(field)
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    result = tuple(value)
    if not result or not all(isinstance(item, str) and item.strip() for item in result):
        raise ValueError(f"{field} must contain non-empty strings")
    return result


def _parse_experiment(record: Mapping[str, Any]) -> Experiment:
    model = _required_text(record, "model")
    if model not in {"double", "triple", "golfer"}:
        raise ValueError(f"unsupported companion model: {model}")
    return Experiment(
        id=_required_text(record, "id"),
        title=_required_text(record, "title"),
        model=model,
        purpose=_required_text(record, "purpose"),
        hypothesis=_required_text(record, "hypothesis"),
        falsifier=_required_text(record, "falsifier"),
        workflow=_required_text_tuple(record, "workflow"),
        tips=_required_text_tuple(record, "tips"),
        observables=_required_text_tuple(record, "observables"),
        limitations=_required_text_tuple(record, "limitations"),
    )


def _parse_term(record: Mapping[str, Any]) -> GlossaryTerm:
    return GlossaryTerm(
        id=_required_text(record, "id"),
        term=_required_text(record, "term"),
        definition=_required_text(record, "definition"),
        plain_language=_required_text(record, "plain_language"),
        units=_required_text(record, "units"),
        caution=_required_text(record, "caution"),
    )


def _require_unique_ids(records: tuple[Experiment | GlossaryTerm, ...], name: str) -> None:
    identifiers = [record.id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"{name} IDs must be unique")


def load_companion_catalog() -> CompanionCatalog:
    """Load and validate the canonical companion catalog."""
    resource = files("double_pendulum_golf").joinpath(_CATALOG_RESOURCE)
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("companion catalog root must be an object")
    experiments = tuple(_parse_experiment(item) for item in payload["experiments"])
    glossary = tuple(_parse_term(item) for item in payload["glossary"])
    _require_unique_ids(experiments, "experiment")
    _require_unique_ids(glossary, "glossary")
    return CompanionCatalog(
        schema_version=_required_text(payload, "schema_version"),
        title=_required_text(payload, "title"),
        scientific_status=_required_text(payload, "scientific_status"),
        experiments=experiments,
        glossary=glossary,
    )


def search_glossary(catalog: CompanionCatalog, query: str) -> tuple[GlossaryTerm, ...]:
    """Return terms matching a case-insensitive reader query."""
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    needle = query.strip().casefold()
    if not needle:
        return catalog.glossary
    return tuple(
        term
        for term in catalog.glossary
        if needle
        in " ".join((term.term, term.definition, term.plain_language, term.caution)).casefold()
    )


def build_run_manifest(
    experiment_id: str,
    parameters: Mapping[str, Any],
    units: Mapping[str, str],
    model_version: str,
) -> Mapping[str, Any]:
    """Build immutable, self-describing metadata for an exported run."""
    if not isinstance(experiment_id, str):
        raise TypeError("experiment_id must be a string")
    if not experiment_id.strip():
        raise ValueError("experiment_id must not be empty")
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping")
    if not isinstance(units, Mapping):
        raise TypeError("units must be a mapping")
    if not isinstance(model_version, str):
        raise TypeError("model_version must be a string")
    if not model_version.strip():
        raise ValueError("model_version must not be empty")
    if not all(isinstance(key, str) for key in parameters):
        raise TypeError("parameter names must be strings")
    if not all(
        isinstance(key, str) and isinstance(value, str) for key, value in units.items()
    ):
        raise TypeError("units must map strings to strings")
    return {
        "schema_version": _RUN_SCHEMA_VERSION,
        "scientific_status": "exploratory_model_output",
        "experiment_id": experiment_id,
        "model_version": model_version,
        "parameters": dict(parameters),
        "units": dict(units),
    }
