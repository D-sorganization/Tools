"""Regression tests for assessment-generator script topology."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
REMOVED_GENERATORS = (
    "generate_assessments.py",
    "generate_fresh_assessments.py",
)


def test_comprehensive_assessment_generator_is_canonical() -> None:
    """Only the comprehensive assessment generator remains executable."""
    assert (SCRIPTS_DIR / "generate_comprehensive_assessment.py").is_file()
    for script_name in REMOVED_GENERATORS:
        assert not (SCRIPTS_DIR / script_name).exists()


def test_live_tree_has_no_references_to_removed_generators() -> None:
    """Live automation and docs must not point at removed generators."""
    live_roots = (
        REPO_ROOT / ".github",
        REPO_ROOT / "docs" / "development",
        REPO_ROOT / "scripts",
        REPO_ROOT / "tests",
    )
    current_file = Path(__file__).resolve()
    for root in live_roots:
        for path in root.rglob("*"):
            if path == current_file or path.suffix not in {
                ".md",
                ".py",
                ".yaml",
                ".yml",
            }:
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            for script_name in REMOVED_GENERATORS:
                assert script_name not in content, f"{path} references {script_name}"
