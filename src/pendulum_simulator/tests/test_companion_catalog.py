"""Contract tests for the proximal-distal companion catalog."""

from __future__ import annotations

import json

import pytest

from double_pendulum_golf.companion_catalog import (
    build_run_manifest,
    load_companion_catalog,
    search_glossary,
)


def test_catalog_covers_each_interactive_model() -> None:
    catalog = load_companion_catalog()

    assert {experiment.model for experiment in catalog.experiments} == {
        "double",
        "triple",
        "golfer",
    }


def test_experiments_expose_falsifiability_and_learning_contracts() -> None:
    catalog = load_companion_catalog()

    assert len(catalog.experiments) >= 6
    for experiment in catalog.experiments:
        assert experiment.hypothesis
        assert experiment.falsifier
        assert experiment.workflow
        assert experiment.tips
        assert experiment.observables
        assert experiment.limitations


def test_glossary_search_matches_term_and_definition() -> None:
    catalog = load_companion_catalog()

    matches = search_glossary(catalog, "counterfactual")

    assert {match.id for match in matches} >= {
        "forward-counterfactual",
        "ztcf",
    }


def test_run_manifest_is_json_serializable_and_self_describing() -> None:
    manifest = build_run_manifest(
        experiment_id="double-passive-transfer",
        parameters={"wrist_torque": -8.0, "duration": 0.8},
        units={"wrist_torque": "N m", "duration": "s"},
        model_version="tools-pendulum-0.1.0",
    )

    encoded = json.dumps(manifest)

    assert "double-passive-transfer" in encoded
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["scientific_status"] == "exploratory_model_output"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("experiment_id", "", ValueError),
        ("parameters", [], TypeError),
        ("units", [], TypeError),
        ("model_version", "", ValueError),
    ],
)
def test_run_manifest_rejects_invalid_public_inputs(
    field: str, value: object, error: type[Exception]
) -> None:
    kwargs: dict[str, object] = {
        "experiment_id": "double-passive-transfer",
        "parameters": {"duration": 0.8},
        "units": {"duration": "s"},
        "model_version": "tools-pendulum-0.1.0",
    }
    kwargs[field] = value

    with pytest.raises(error):
        build_run_manifest(**kwargs)  # type: ignore[arg-type]
