from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]


def _project_metadata() -> dict[str, object]:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_dead_dwsim_console_script_is_not_advertised() -> None:
    """Console scripts must point at importable package modules."""
    metadata = _project_metadata()
    scripts = metadata["project"].get("scripts", {})  # type: ignore[union-attr]

    assert "dwsim-model" not in scripts


def test_shared_compatibility_module_is_packaged() -> None:
    """The bare compatibility shim must be present in built wheels."""
    metadata = _project_metadata()
    setuptools = metadata["tool"]["setuptools"]  # type: ignore[index]
    py_modules = set(setuptools["py-modules"])  # type: ignore[index]
    package_dir = setuptools["package-dir"]  # type: ignore[index]

    assert "compatibility" in py_modules
    assert package_dir["compatibility"] == "src/shared/python"  # type: ignore[index]


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
