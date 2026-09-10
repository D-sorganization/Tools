"""Navigation measurements use externally specified expected paths and consumers."""

from pathlib import Path

import pytest

from agent_context.catalog import CatalogError
from agent_context.evaluation import evaluate


def test_task_evaluation_measures_expected_evidence(repository: Path) -> None:
    tasks = [
        {
            "query": "convert",
            "component": "provider",
            "source": "src/provider.py",
            "consumer": "consumer",
        }
    ]
    result = evaluate(repository, tasks)
    assert result["passed"]
    assert result["correct"] == 1
    assert result["tasks"][0]["characters"] <= 16000
    assert result["tasks"][0]["elapsed_ms"] >= 0


def test_wrong_expectation_cannot_be_reported_as_success(repository: Path) -> None:
    result = evaluate(
        repository,
        [{"query": "conversion", "component": "missing", "source": "src/missing.py"}],
    )
    assert not result["passed"]
    assert result["correct"] == 0


def test_evaluation_rejects_checkout_changes(
    repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_context import evaluation

    original = evaluation._task

    def changing_task(service: object, task: dict) -> dict:
        result = original(service, task)
        (repository / "src/provider.py").write_text("def convert(): return 42\n")
        return result

    monkeypatch.setattr(evaluation, "_task", changing_task)
    with pytest.raises(CatalogError, match="changed"):
        evaluate(
            repository,
            [
                {
                    "query": "convert",
                    "component": "provider",
                    "source": "src/provider.py",
                }
            ],
        )
