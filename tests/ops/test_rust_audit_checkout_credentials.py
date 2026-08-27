from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_STANDARD = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"


def test_rust_audit_checkout_does_not_persist_repository_credentials() -> None:
    """Public RustSec fetches must not inherit repo-scoped checkout auth."""
    workflow = yaml.safe_load(CI_STANDARD.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["rust-quality-gate"]["steps"]
    checkout = next(
        step
        for step in steps
        if str(step.get("uses", "")).startswith("actions/checkout@")
    )
    audit = next(
        step for step in steps if step.get("name") == "Security Audit (cargo-audit)"
    )

    assert checkout["with"]["persist-credentials"] is False
    assert "cargo audit" in audit["run"]
    assert "GIT_CONFIG_GLOBAL=/dev/null" in audit["run"]
    assert "GIT_CONFIG_NOSYSTEM=1" in audit["run"]
    assert "GIT_TERMINAL_PROMPT=0" in audit["run"]
