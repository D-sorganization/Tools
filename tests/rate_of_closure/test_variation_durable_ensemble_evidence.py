"""Portable durable-summary evidence shared by desktop and web clients."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from rate_of_closure.variation.durable_ensemble_chunks import DurableEnsembleArchive
from rate_of_closure.variation.durable_ensemble_evidence import (
    DURABLE_ENSEMBLE_EVIDENCE_SCHEMA,
    durable_ensemble_evidence,
    durable_ensemble_evidence_from_json,
    durable_ensemble_evidence_to_json,
)
from rate_of_closure.variation.plot_labels import OUTPUT_UNITS
from rate_of_closure.variation.simulation_types import ALL_OUTPUT_NAMES
from rate_of_closure.variation.streaming_ensemble_analysis import (
    DurableEnsembleLayout,
    DurableEnsembleSummary,
    StreamingOutputMoments,
)
from shared.python.contracts import ContractViolationError

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
_GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "src/rate_of_closure/web/src/model/__fixtures__"
    / "durable_ensemble_evidence_golden_v1.json"
)


def _summary(tmp_path: Path) -> DurableEnsembleSummary:
    moments = tuple(
        StreamingOutputMoments(name, OUTPUT_UNITS[name], 2, 1.25, 0.5)
        for name in ALL_OUTPUT_NAMES
    )
    archive = DurableEnsembleArchive(
        tmp_path.resolve(), "a" * 64, "in_progress", 5, 3, 1, 2, None
    )
    return DurableEnsembleSummary(
        archive,
        DurableEnsembleLayout(
            101,
            ("swing.pivot", "swing.clubhead.reference"),
            "app_frame:x_target,y_up,z_right",
        ),
        3,
        {"evaluated_hit": 2, "evaluated_no_impact": 0, "numerical_failure": 1},
        {"RuntimeError": 1},
        moments,
    )


def test_evidence_omits_private_path_and_round_trips_exactly(tmp_path: Path) -> None:
    evidence = durable_ensemble_evidence(_summary(tmp_path))
    text = durable_ensemble_evidence_to_json(evidence)
    restored = durable_ensemble_evidence_from_json(text)

    assert restored == evidence
    assert restored.schema_version == DURABLE_ENSEMBLE_EVIDENCE_SCHEMA
    assert restored.archive.header_sha256 == "a" * 64
    assert restored.analysis.coordinate_frame == "app_frame:x_target,y_up,z_right"
    assert restored.analysis.method_id == "incremental-welford-sample-moments/v1"
    assert str(tmp_path.resolve()) not in text
    assert "directory" not in json.loads(text)["archive"]


def test_python_builder_and_react_fixture_are_identical(tmp_path: Path) -> None:
    expected = json.loads(
        durable_ensemble_evidence_to_json(durable_ensemble_evidence(_summary(tmp_path)))
    )

    assert json.loads(_GOLDEN.read_text(encoding="utf-8")) == expected
    assert durable_ensemble_evidence_from_json(
        _GOLDEN.read_text(encoding="utf-8")
    ) == durable_ensemble_evidence(_summary(tmp_path))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda item: item.update(extra=True), "fields"),
        (
            lambda item: item["archive"].update(analyzed_trial_count=4),
            "status counts",
        ),
        (
            lambda item: item["output_moments"][0].update(unit="ft"),
            "canonical",
        ),
        (
            lambda item: item["analysis"].update(coordinate_frame="private.frame"),
            "frame",
        ),
    ],
)
def test_evidence_parser_rejects_untrusted_drift(
    tmp_path: Path, mutation: Callable[[dict[str, object]], None], message: str
) -> None:
    item = json.loads(
        durable_ensemble_evidence_to_json(durable_ensemble_evidence(_summary(tmp_path)))
    )
    mutation(item)

    with pytest.raises(ContractViolationError, match=message):
        durable_ensemble_evidence_from_json(json.dumps(item))
