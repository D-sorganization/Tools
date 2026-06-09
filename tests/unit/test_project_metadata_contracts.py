from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _project_metadata() -> dict[str, object]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_dead_dwsim_console_script_is_not_advertised() -> None:
    """Console scripts must point at importable package modules."""
    metadata = _project_metadata()
    scripts = metadata["project"].get("scripts", {})  # type: ignore[union-attr]

    assert "dwsim-model" not in scripts


def test_readme_python_badge_matches_package_requires_python() -> None:
    metadata = _project_metadata()
    requires_python = metadata["project"]["requires-python"]  # type: ignore[index]
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert requires_python == ">=3.11"
    assert "python-3.11+" in readme
    assert "Version **3.11+** required" in readme
    assert not re.search(r"Python 3\.10|python-3\.10|3\.10\+", readme)


def test_current_user_docs_do_not_claim_root_python_310_support() -> None:
    current_docs = [
        REPO_ROOT / "docs" / "ARCHITECTURE_OVERVIEW.md",
        REPO_ROOT / "docs" / "ONBOARDING.md",
        REPO_ROOT / "docs" / "USER_MANUAL.md",
        REPO_ROOT / "docs" / "development" / "QUICKSTART.md",
        REPO_ROOT / "docs" / "tutorials" / "quick_start.md",
    ]

    for path in current_docs:
        content = path.read_text(encoding="utf-8")
        assert not re.search(r"Python 3\.10|3\.10\+", content), path
