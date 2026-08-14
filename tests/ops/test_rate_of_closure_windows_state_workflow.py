"""Static governance contract for native Windows authority-state CI."""

from __future__ import annotations

from pathlib import Path

import yaml

_WORKFLOW = Path(".github/workflows/rate-of-closure-windows-state-security.yml")


def test_windows_state_workflow_uses_only_restricted_runner_and_public_fetch() -> None:
    source = _WORKFLOW.read_text(encoding="utf-8")
    workflow = yaml.safe_load(source)
    assert workflow["permissions"] == {"contents": "read"}
    assert set(workflow["jobs"]) == {"qualify"}
    job = workflow["jobs"]["qualify"]
    assert job["runs-on"] == [
        "self-hosted",
        "Windows",
        "X64",
        "d-sorg-windows-security",
    ]
    assert job["strategy"]["matrix"]["python"] == ["3.11", "3.12"]
    assert "actions/checkout" not in source
    assert "credential.helper=" in source
    assert "pull_request.head.sha || github.sha" in source
    assert "ROC_REQUIRE_WINDOWS_SYMLINK_TEST" in source


def test_windows_state_workflow_qualifies_native_and_installed_contracts() -> None:
    source = _WORKFLOW.read_text(encoding="utf-8")
    for required in (
        "test_authority_state_security.py",
        "test_regional_ground_authority_store.py",
        "npm run type-check",
        "npm run lint",
        "npm run build",
        "python -m build --wheel",
        "qualify_windows_authority_state_install.py",
        "unrelated ü cwd",
        "[rate-of-closure-web]",
    ):
        assert required in source
    assert "upload-artifact" not in source
    assert "secrets." not in source
