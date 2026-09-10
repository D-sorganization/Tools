"""Python consumer for the shared regional scalar-ensemble golden fixture.

The TypeScript half of this contract lives in
`src/rate_of_closure/web/src/model/regionalGroundResultImport.test.ts`; both
runtimes must accept the same fixture bytes and preserve the same Python-owned
metadata, ordering, digests, and typed nulls (issue #4560).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rate_of_closure.variation.scalar_ensemble_wire import (
    scalar_ensemble_dataset_from_wire,
)
from shared.python.swing_sim.ground.strict_json import strict_json_object

pytestmark = [pytest.mark.unit, pytest.mark.contract, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).parents[2].resolve()
_FIXTURE_PATH = (
    _REPO_ROOT
    / "src"
    / "rate_of_closure"
    / "web"
    / "src"
    / "model"
    / "__fixtures__"
    / "regional_ground_scalar_ensemble_golden_v1.json"
)


@pytest.fixture(scope="module")
def fixture_payload() -> dict[str, Any]:
    """Load the shared golden fixture exactly as the TypeScript side does."""
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_python_wire_parser_accepts_the_shared_golden_bytes(
    fixture_payload: dict[str, Any],
) -> None:
    """The strict untrusted-value parser must accept the fixture verbatim."""
    dataset = scalar_ensemble_dataset_from_wire(fixture_payload)

    assert dataset.schema_version == fixture_payload["schema_version"]
    assert dataset.result_id == fixture_payload["result_id"]
    assert dataset.provenance.adapter_id == fixture_payload["provenance"]["adapter_id"]


def test_python_preserves_python_owned_metadata_and_ordering(
    fixture_payload: dict[str, Any],
) -> None:
    """Definitions must round-trip in Python-authored order."""
    dataset = scalar_ensemble_dataset_from_wire(fixture_payload)

    assert [category.key for category in dataset.categories] == [
        entry["key"] for entry in fixture_payload["categories"]
    ]
    assert [category.label for category in dataset.categories] == [
        entry["label"] for entry in fixture_payload["categories"]
    ]
    assert [cohort.key for cohort in dataset.cohorts] == [
        entry["key"] for entry in fixture_payload["cohorts"]
    ]
    assert [row.trial_index for row in dataset.rows] == [0, 1, 2, 3]
    assert [row.cohort for row in dataset.rows] == [
        "complete",
        "partial",
        "failed",
        "unavailable",
    ]


def test_python_preserves_typed_nulls_and_attributes(
    fixture_payload: dict[str, Any],
) -> None:
    """Typed nulls must stay distinct from zero and attributes must survive."""
    dataset = scalar_ensemble_dataset_from_wire(fixture_payload)

    first = dataset.rows[0]
    assert first.attributes is not None
    assert (
        first.attributes["ground_model_id"]
        == "tools-ground-impact-bounce+tools-ground-skid-roll"
    )
    variation_input = first.attributes.get("variation_input_sha256")
    assert variation_input is not None and len(variation_input) == 64

    for row in dataset.rows[1:]:
        value = row.values["metric.total_distance"]
        assert value is None
        assert value is not 0  # noqa: F632 - typed-null contract, not identity


def test_python_rejects_forged_duplicate_keys(
    fixture_payload: dict[str, Any],
) -> None:
    """A duplicate-key forgery must fail closed, matching the TS importer.

    Duplicate rejection happens at the text-to-object boundary
    (:func:`strict_json_object`), the same envelope layer the regional-ground
    execution result uses before the wire parser sees the mapping.
    """
    text = json.dumps(fixture_payload)
    forged = text.replace('"result_id":', '"result_id":"forged","result_id":', 1)

    with pytest.raises(ValueError, match="duplicate"):
        document = strict_json_object(forged)
        scalar_ensemble_dataset_from_wire(document)
