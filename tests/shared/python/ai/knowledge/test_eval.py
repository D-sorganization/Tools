"""Tests for golden Q&A evaluation of knowledge packs (Tools #5347)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from shared.python.ai.knowledge import (
    KnowledgePack,
    build_pack,
    manifest_from_dict,
)
from shared.python.ai.knowledge.eval import (
    EvalSummary,
    GoldenQACase,
    evaluate_pack,
    load_golden_set,
)
from shared.python.ai.knowledge.eval import (
    main as eval_main,
)


@pytest.fixture
def eval_pack(corpus: dict[str, Path], tmp_path: Path) -> KnowledgePack:
    manifest = manifest_from_dict(
        {
            "id": "findings",
            "title": "Findings",
            "sources": [
                {
                    "repo": "AffineDrift",
                    "authority": "published",
                    "include": ["articles/**/*.qmd"],
                    "exclude": ["**/*-bibliography.md"],
                },
                {
                    "repo": "UpstreamDrift",
                    "authority": "findings",
                    "include": ["docs/research/*", "docs/assessments/*"],
                },
            ],
            "status_overrides": {
                "docs/assessments/bounce_old.md": "superseded",
            },
        }
    )
    pack_path = tmp_path / "findings.pack"
    build_pack(manifest, corpus, pack_path)
    return KnowledgePack.open(pack_path)


@pytest.fixture
def golden_yaml_path(tmp_path: Path) -> Path:
    cases = [
        {
            "id": "case-pelvis",
            "question": "peak pelvis speed torso timing",
            "expected_source": "articles/energy/*.qmd",
            "must_not_source": ["**/*bibliography*"],
            "category": "biomechanics",
        },
        {
            "id": "case-filter",
            "question": "Butterworth filter segment chain",
            "expected_source": "docs/research/*.tex",
            "category": "signal_processing",
        },
        {
            "id": "case-bounce",
            "question": "bounce digs deeper model",
            "expected_source": "docs/assessments/bounce_new.md",
            "must_not_source": ["docs/assessments/bounce_old.md"],
            "category": "equipment",
        },
    ]
    path = tmp_path / "golden.eval.yml"
    path.write_text(yaml.safe_dump(cases), encoding="utf-8")
    return path


def test_load_golden_set_valid(golden_yaml_path: Path) -> None:
    cases = load_golden_set(golden_yaml_path)
    assert len(cases) == 3
    assert cases[0].id == "case-pelvis"
    assert cases[0].question == "peak pelvis speed torso timing"
    assert cases[0].expected_source == ("articles/energy/*.qmd",)
    assert cases[0].must_not_source == ("**/*bibliography*",)
    assert cases[0].category == "biomechanics"


def test_load_golden_set_mapping_format(tmp_path: Path) -> None:
    data = {
        "cases": [
            {
                "question": "Where does elastic energy store in shaft?",
                "expected_source": ["articles/energy/energy.qmd"],
            }
        ]
    }
    path = tmp_path / "mapped.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    cases = load_golden_set(path)
    assert len(cases) == 1
    assert cases[0].expected_source == ("articles/energy/energy.qmd",)
    assert cases[0].must_not_source == ()


def test_load_golden_set_rejects_invalid(tmp_path: Path) -> None:
    # Empty question
    path1 = tmp_path / "bad1.yml"
    path1.write_text(
        yaml.safe_dump([{"question": "", "expected_source": "a.md"}]), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="question"):
        load_golden_set(path1)

    # Missing expected_source
    path2 = tmp_path / "bad2.yml"
    path2.write_text(yaml.safe_dump([{"question": "some q"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="expected_source"):
        load_golden_set(path2)


def test_evaluate_pack_metrics(
    eval_pack: KnowledgePack, golden_yaml_path: Path
) -> None:
    cases = load_golden_set(golden_yaml_path)
    summary = evaluate_pack(eval_pack, cases, k=8)

    assert isinstance(summary, EvalSummary)
    assert summary.total_cases == 3
    assert summary.passed_cases == 3
    assert summary.failed_cases == 0
    assert summary.recall_at_k == 1.0
    assert summary.mrr > 0.5
    assert summary.must_not_violation_count == 0

    # Test category breakdown
    assert "biomechanics" in summary.category_metrics
    assert summary.category_metrics["biomechanics"]["passed"] == 1

    # Test serialization
    as_dict = summary.to_dict()
    assert as_dict["recall_at_k"] == 1.0
    assert len(as_dict["results"]) == 3

    markdown = summary.to_markdown()
    assert "Evaluation Report" in markdown
    assert "Recall@8" in markdown


def test_evaluate_pack_must_not_source_violation(eval_pack: KnowledgePack) -> None:
    # A case where must_not_source matches a retrieved hit
    cases = [
        GoldenQACase(
            id="bad-must-not",
            question="Peak pelvis speed timing",
            expected_source=("articles/energy/*.qmd",),
            must_not_source=("articles/energy/energy.qmd",),  # It will retrieve this!
        )
    ]
    summary = evaluate_pack(eval_pack, cases, k=8)
    assert summary.total_cases == 1
    assert summary.passed_cases == 0
    assert summary.must_not_violation_count == 1
    assert not summary.results[0].passed


def test_evaluate_pack_unmatched_question(eval_pack: KnowledgePack) -> None:
    cases = [
        GoldenQACase(
            id="miss",
            question="quantum teleportation entangled photon",
            expected_source=("articles/quantum/*.qmd",),
        )
    ]
    summary = evaluate_pack(eval_pack, cases, k=8)
    assert summary.recall_at_k == 0.0
    assert summary.mrr == 0.0
    assert summary.passed_cases == 0


def test_eval_cli_success(
    eval_pack: KnowledgePack, golden_yaml_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = eval_main(
        [str(eval_pack._path), str(golden_yaml_path), "-k", "8", "--min-recall", "0.8"]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "Recall@8" in captured.out


def test_eval_cli_fails_on_threshold(
    eval_pack: KnowledgePack, golden_yaml_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Require 100% when there is an unmatchable case
    bad_yaml = golden_yaml_path.parent / "unmatchable.yml"
    bad_yaml.write_text(
        yaml.safe_dump(
            [
                {
                    "question": "unmatched photon entangled qubit",
                    "expected_source": "articles/nowhere/*.md",
                }
            ]
        ),
        encoding="utf-8",
    )
    code = eval_main([str(eval_pack._path), str(bad_yaml), "--min-recall", "0.5"])
    assert code == 1
    captured = capsys.readouterr()
    assert "FAIL" in captured.err


def test_eval_recorded_baseline_does_not_regress(
    eval_pack: KnowledgePack, golden_yaml_path: Path
) -> None:
    """CI baseline gate: recall@8 must not regress below recorded floor.

    Recorded floor is 1.0 on golden set.
    """
    cases = load_golden_set(golden_yaml_path)
    summary = evaluate_pack(eval_pack, cases, k=8)
    RECORDED_BASELINE_RECALL = 1.0
    assert summary.recall_at_k >= RECORDED_BASELINE_RECALL, (
        f"Recall@8 regressed: {summary.recall_at_k} < {RECORDED_BASELINE_RECALL}"
    )
