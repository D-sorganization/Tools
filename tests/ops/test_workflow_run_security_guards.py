"""Security contracts for agent workflows (Tools #4923, #4464, #4502).

* Every job in a workflow triggered by ``workflow_run`` must refuse to act on
  a run whose head repository is not this repository. ``workflow_run`` fires
  with the *base* repository's permissions and secrets even when the
  triggering run came from a fork PR, so a write-capable job (``contents:
  write`` / ``pull-requests: write``, or a reusable-workflow call whose callee
  has them) that checks out ``workflow_run.head_branch`` and pushes is the
  classic pwn-request. The guard is the documented
  ``github.event.workflow_run.head_repository.full_name == github.repository``.
* CodeQL must be enabled for python and javascript on PRs and on a schedule.
* The pip-audit step must not carry an undated/unexpired vulnerability ignore.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
SAME_REPO_GUARD = (
    "github.event.workflow_run.head_repository.full_name == github.repository"
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), path
    return data


def _triggers(workflow: dict) -> dict:
    on = workflow.get("on", workflow.get(True, {}))
    return on if isinstance(on, dict) else {}


def _workflow_run_workflows() -> list[Path]:
    return sorted(
        path
        for path in WORKFLOWS.glob("*.yml")
        if "workflow_run" in _triggers(_load(path))
    )


def test_workflow_run_workflows_are_the_known_set() -> None:
    names = {path.name for path in _workflow_run_workflows()}
    assert names == {
        "Jules-Control-Tower.yml",
        "Jules-PR-AutoFix.yml",
    }, "new workflow_run consumer: extend the guard and this list deliberately"


@pytest.mark.parametrize("path", _workflow_run_workflows(), ids=lambda p: p.name)
def test_every_job_of_a_workflow_run_workflow_carries_the_same_repo_guard(
    path: Path,
) -> None:
    workflow = _load(path)
    for job_id, job in workflow["jobs"].items():
        condition = str(job.get("if", ""))
        assert SAME_REPO_GUARD in condition, (
            f"{path.name}: job {job_id!r} runs on workflow_run without the "
            f"same-repository guard ({SAME_REPO_GUARD})"
        )


def test_pr_autofix_guard_covers_the_write_capable_job() -> None:
    workflow = _load(WORKFLOWS / "Jules-PR-AutoFix.yml")
    assert workflow["permissions"]["contents"] == "write"
    fix = workflow["jobs"]["iterative-fix"]
    assert SAME_REPO_GUARD in fix["if"]
    # The failure-only gate stays; the guard is added, not substituted.
    assert "github.event.workflow_run.conclusion == 'failure'" in fix["if"]


def test_control_tower_dispatch_jobs_are_guarded_before_calling_writers() -> None:
    workflow = _load(WORKFLOWS / "Jules-Control-Tower.yml")
    callers = {
        job_id: job
        for job_id, job in workflow["jobs"].items()
        if str(job.get("uses", "")).startswith("./.github/workflows/")
    }
    assert callers, "Control-Tower dispatches reusable workflows"
    for job_id, job in callers.items():
        assert SAME_REPO_GUARD in job["if"], job_id
        assert "needs.triage.outputs.target" in job["if"], job_id


def test_codeql_is_enabled_for_python_and_javascript() -> None:
    assert not (WORKFLOWS / "codeql-analysis.yml.disabled").exists()
    workflow = _load(WORKFLOWS / "codeql-analysis.yml")
    triggers = _triggers(workflow)
    assert "pull_request" in triggers
    assert "schedule" in triggers
    analyze = workflow["jobs"]["analyze"]
    assert analyze["strategy"]["matrix"]["language"] == [
        "python",
        "javascript-typescript",
    ]
    assert analyze["permissions"]["security-events"] == "write"
    uses = [str(step.get("uses", "")) for step in analyze["steps"]]
    assert any(u.startswith("github/codeql-action/init@") for u in uses)
    assert any(u.startswith("github/codeql-action/analyze@") for u in uses)
    assert "timeout-minutes" in analyze


def test_pip_audit_ignores_carry_a_dated_justification_and_expiry() -> None:
    text = (WORKFLOWS / "ci-standard.yml").read_text(encoding="utf-8")
    step = text.split("- name: Security Scan (pip-audit)", maxsplit=1)[1].split(
        "- name:", maxsplit=1
    )[0]
    # Only real command lines count; the step comment documents the rule and
    # spells the flag out.
    command_lines = "\n".join(
        line for line in step.splitlines() if not line.strip().startswith("#")
    )
    ignores = re.findall(r"--ignore-vuln\s+(\S+)", command_lines)
    for vuln in ignores:
        dated = re.search(
            rf"{re.escape(vuln)}.*(added|since|dated)\s+20\d\d-\d\d-\d\d",
            step,
            re.IGNORECASE,
        )
        assert dated, f"{vuln}: pip-audit ignore needs a dated justification comment"
        expiry = re.search(
            rf"{re.escape(vuln)}.*(expires?|until)\s+20\d\d-\d\d-\d\d",
            step,
            re.IGNORECASE,
        )
        assert expiry, f"{vuln}: pip-audit ignore needs an expiry date"
