"""Exercise CI checkout history against a multi-commit push diff."""

import subprocess
from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/ci-standard.yml"


def test_quality_gate_can_diff_multi_commit_push(tmp_path: Path) -> None:
    """The push before SHA must survive the configured checkout depth."""
    source = tmp_path / "source"
    source.mkdir()

    def git(*args: str, cwd: Path = source) -> str:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    for index in range(4):
        (source / "changed.py").write_text(f"VALUE = {index}\n")
        git("add", ".")
        git("commit", "-qm", f"Change {index}")
        if index == 0:
            base = git("rev-parse", "HEAD")

    workflow = yaml.safe_load(WORKFLOW.read_text())
    checkout = next(
        step
        for step in workflow["jobs"]["quality-gate"]["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout@")
    )
    depth = checkout.get("with", {}).get("fetch-depth", 1)
    clone = tmp_path / "checkout"
    depth_args = ["--depth", str(depth)] if depth else []
    git("clone", "-q", *depth_args, source.as_uri(), str(clone))
    assert git("diff", "--name-only", base, "HEAD", cwd=clone) == "changed.py"
