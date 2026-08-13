import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/check_local_only_workflows.py").resolve()


def _run_guard(
    tmp_path: Path, workflow: str, filename: str = "ci-standard.yml"
) -> subprocess.CompletedProcess[str]:
    workflow_dir = tmp_path / ".github" / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    (workflow_dir / filename).write_text(workflow, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )


def test_guard_allows_only_hosted_ci_standard_quality_gate(tmp_path: Path) -> None:
    allowed = _run_guard(
        tmp_path,
        "jobs:\n  quality-gate:\n    runs-on: ubuntu-24.04\n",
    )
    rejected = _run_guard(
        tmp_path,
        "jobs:\n  integration:\n    runs-on: ubuntu-24.04\n",
    )

    assert allowed.returncode == 0
    assert rejected.returncode == 1


def test_guard_allows_only_ephemeral_fork_playwright_job(tmp_path: Path) -> None:
    allowed = _run_guard(
        tmp_path,
        "jobs:\n  fork-production-worker-e2e:\n    runs-on: ubuntu-latest\n",
        "rate-web-playwright.yml",
    )
    rejected = _run_guard(
        tmp_path,
        "jobs:\n  trusted-production-worker-e2e:\n    runs-on: ubuntu-latest\n",
        "rate-web-playwright.yml",
    )

    assert allowed.returncode == 0
    assert rejected.returncode == 1
