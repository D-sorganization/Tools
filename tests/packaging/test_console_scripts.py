"""Packaging contract test for every declared console script.

Regression guard for issue #3253: a `[project.scripts]` entry pointed at a
``dwsim_model`` package that does not exist anywhere in the source tree, so an
installed distribution exposed a ``dwsim-model`` command that failed on first
invocation. This test enumerates *every* declared console-script entry point
and asserts its target module imports and exposes the named callable, so a
broken or stale script declaration can no longer ship unnoticed.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _console_scripts() -> dict[str, str]:
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = config["project"]["scripts"]
    assert isinstance(scripts, dict)
    return {str(name): str(target) for name, target in scripts.items()}


def test_at_least_one_console_script_declared() -> None:
    assert _console_scripts(), "expected at least one [project.scripts] entry"


@pytest.mark.contract
@pytest.mark.parametrize(
    ("name", "target"),
    sorted(_console_scripts().items()),
    ids=sorted(_console_scripts()),
)
def test_console_script_target_is_importable(name: str, target: str) -> None:
    """Every ``console = module:attr`` target must import and expose ``attr``.

    ``console = "pkg.mod:func"`` is the entry-point spec setuptools uses to
    generate the executable. If the module cannot be imported or ``func`` is
    missing, the installed command fails immediately on invocation.
    """
    assert ":" in target, f"{name!r} entry point {target!r} must be 'module:callable'"
    module_path, _, attr = target.partition(":")

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"console script {name!r} target module {module_path!r} is missing: {exc}"
        )

    obj = module
    for part in attr.split("."):
        assert hasattr(obj, part), (
            f"console script {name!r} target {target!r}: "
            f"{type(obj).__name__} has no attribute {part!r}"
        )
        obj = getattr(obj, part)

    assert callable(obj), f"console script {name!r} target {target!r} is not callable"
