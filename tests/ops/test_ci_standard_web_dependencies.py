from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_STANDARD = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"


def test_ci_standard_installs_fastapi_multipart_parser() -> None:
    workflow = CI_STANDARD.read_text(encoding="utf-8")
    fastapi_install_lines = [
        line.strip()
        for line in workflow.splitlines()
        if line.strip().startswith("python -m pip install fastapi")
    ]

    assert fastapi_install_lines
    assert all("python-multipart" in line.split() for line in fastapi_install_lines)
