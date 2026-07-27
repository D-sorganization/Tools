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


def test_standalone_sidekick_runtime_dependency_is_declared() -> None:
    """The canonical standalone profile store must install its path provider."""
    metadata = _project_metadata()
    dependencies = metadata["project"]["dependencies"]  # type: ignore[index]

    assert any(
        dependency.partition(">=")[0].casefold() == "platformdirs"
        for dependency in dependencies
    )


def test_standalone_sidekick_console_script_is_declared() -> None:
    """Installed Tools artifacts expose the supported standalone launcher."""
    metadata = _project_metadata()
    scripts = metadata["project"]["scripts"]  # type: ignore[index]

    assert scripts["sidekick"] == "sidekick.__main__:main"


def test_shared_modules_are_only_packaged_under_shared_python() -> None:
    """The shared library must not be double-shipped as bare top-level modules."""
    metadata = _project_metadata()
    package_find = metadata["tool"]["setuptools"]["packages"]["find"]  # type: ignore[index]
    pytest_options = metadata["tool"]["pytest"]["ini_options"]  # type: ignore[index]
    mypy_options = metadata["tool"]["mypy"]  # type: ignore[index]
    scripts = metadata["project"].get("scripts", {})  # type: ignore[union-attr]

    assert "src/shared/python" not in package_find["where"]  # type: ignore[index]
    assert "src/shared/python" not in pytest_options["pythonpath"]  # type: ignore[index]
    assert "src/shared/python" not in mypy_options["mypy_path"]  # type: ignore[index]
    assert scripts["urdf-gen"] == "shared.python.model_generation.cli:main"
    assert scripts["generate-pid"] == "shared.python.programmatic_pid.cli:main"
    assert scripts["codemap"] == "shared.python.codemap.cli:main"


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
