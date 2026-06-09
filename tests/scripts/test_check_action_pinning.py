"""Tests for the third-party action SHA-pinning linter (issue #3255)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_action_pinning.py"
)

_SHA = "e18b497796c12c097a38f9edb9d0641fb99eee32"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_action_pinning", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_third_party_tag_is_rejected() -> None:
    module = _load_module()
    assert module.violations_for_reference("Swatinem/rust-cache@v2") is not None


def test_third_party_branch_is_rejected() -> None:
    module = _load_module()
    assert module.violations_for_reference("dtolnay/rust-toolchain@stable") is not None


def test_third_party_sha_is_allowed() -> None:
    module = _load_module()
    assert module.violations_for_reference(f"Swatinem/rust-cache@{_SHA}") is None


def test_first_party_actions_tag_is_allowed() -> None:
    module = _load_module()
    assert module.violations_for_reference("actions/checkout@v6") is None
    assert module.violations_for_reference("github/codeql-action/init@v3") is None


def test_local_and_docker_actions_are_ignored() -> None:
    module = _load_module()
    assert module.violations_for_reference("./.github/actions/local") is None
    assert module.violations_for_reference("docker://alpine:3.20") is None


def test_missing_pin_is_rejected() -> None:
    module = _load_module()
    assert module.violations_for_reference("some/action") is not None


def test_check_file_flags_unpinned_reference(tmp_path: Path) -> None:
    module = _load_module()
    workflow = tmp_path / "wf.yml"
    workflow.write_text(
        "jobs:\n"
        "  build:\n"
        "    steps:\n"
        "      - uses: actions/checkout@v6\n"
        "      - uses: Swatinem/rust-cache@v2\n"
        f"      - uses: dtolnay/rust-toolchain@{_SHA}\n",
        encoding="utf-8",
    )
    problems = module.check_file(workflow)
    assert len(problems) == 1
    assert "Swatinem/rust-cache@v2" in problems[0]


def test_repo_workflows_are_compliant() -> None:
    """The repository's own workflows must pass the pinning policy."""
    module = _load_module()
    workflows_dir = SCRIPT_PATH.resolve().parents[1] / ".github" / "workflows"
    assert module.main([str(workflows_dir)]) == 0


def test_main_returns_nonzero_on_violation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_module()
    workflow = tmp_path / "wf.yml"
    workflow.write_text(
        "jobs:\n  b:\n    steps:\n      - uses: evil/action@v1\n",
        encoding="utf-8",
    )
    assert module.main([str(tmp_path)]) == 1
    assert "evil/action@v1" in capsys.readouterr().err
