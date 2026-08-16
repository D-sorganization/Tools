"""Workflow contracts for the standard-library-only tools manifest gate."""

from pathlib import Path

import yaml

WORKFLOW = Path(".github/workflows/check-tools-manifest.yml")


def _manifest_steps() -> list[dict[str, object]]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["check-manifest"]["steps"]
    assert isinstance(steps, list)
    return steps


def _step(name: str) -> dict[str, object]:
    return next(step for step in _manifest_steps() if step.get("name") == name)


def test_manifest_gate_uses_system_python_without_toolcache_dependency() -> None:
    """The stdlib-only gate must not depend on a mutable setup-python cache."""
    steps = _manifest_steps()

    assert not any(
        str(step.get("uses", "")).startswith("actions/setup-python@")
        for step in steps
    )
    assert all("pip cache purge" not in str(step.get("run", "")) for step in steps)
    assert _step("Generate tools.json and tool_surface_contract.json")["run"] == (
        "python3 scripts/generate_tools_json.py"
    )
    assert _step("Check launcher-backed tool registrations")["run"] == (
        "python3 scripts/check_tools_manifest_layout.py"
    )
    assert "python3 -c" in str(_step("Publish contract summary")["run"])


def test_manifest_workflow_changes_trigger_both_trusted_and_pr_gates() -> None:
    """A workflow-only repair must exercise its own push and PR authorities."""
    source = WORKFLOW.read_text(encoding="utf-8")

    assert source.count('      - ".github/workflows/check-tools-manifest.yml"') == 2
