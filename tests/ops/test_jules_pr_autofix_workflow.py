from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_WORKFLOW = Path(".github/workflows/Jules-PR-AutoFix.yml")
_TOKEN_ENV = "AUTOFIX_GH_TOKEN: ${{ secrets.RUNNER_CHECK_TOKEN || github.token }}"
_TOKEN_REFERENCE = "${{ env.AUTOFIX_GH_TOKEN }}"


def _workflow_text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def test_autofix_token_fallback_is_the_only_runner_check_token_binding() -> None:
    workflow = _workflow_text()

    assert _TOKEN_ENV in workflow
    assert "secrets.RUNNER_CHECK_TOKEN }}" not in workflow


def test_autofix_gh_cli_steps_use_shared_token_fallback() -> None:
    workflow = _workflow_text()
    gh_token_bindings = re.findall(r"\bGH_TOKEN: (?P<value>.+)", workflow)

    assert gh_token_bindings
    assert all(value.strip() == _TOKEN_REFERENCE for value in gh_token_bindings)


def test_autofix_checkout_uses_shared_token_fallback() -> None:
    workflow = _workflow_text()
    checkout_step = workflow[workflow.index("- name: Checkout PR Branch") :]
    token_match = re.search(r"\btoken: (?P<value>.+)", checkout_step)

    assert token_match is not None
    assert token_match.group("value").strip() == _TOKEN_REFERENCE
