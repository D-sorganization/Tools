"""Public-API stability baselines for every vendored shared package (Tools #4920).

``tests/test_sidekick_public_api_stability.py`` guarded only ``sidekick``. The
other packages downstream repos vendor or import -- ``theme``, ``plot_theme``,
``golf_club``, ``swing_sim``, ``launch_monitor``, ``contracts`` and
``safe_eval`` -- had no baseline, so removing a public symbol from
``launch_monitor`` did not fail Tools CI. Each package now has a reviewed
baseline under ``tests/api_baselines/`` and a parametrised test per package.

Policy (CLAUDE.md "Public-API Contract Policy"):

* A symbol is public when a module lists it in ``__all__``; modules without
  ``__all__`` (``contracts.py`` and part of ``theme``) expose every top-level
  function/class/assignment whose name does not start with ``_``.
* Removing a public name, or changing a signature (arguments, defaults,
  return annotation, class bases/public methods) fails these tests.
* A breaking change is allowed only with the baseline bump **in the same PR**
  (``pytest tests/test_shared_package_api_stability.py --regenerate-api-baseline``)
  and a line in the PR description under "Breaking API changes" that names the
  downstream issues (UpstreamDrift / Gasification_Model) tracking the migration;
  the release changelog is generated from PR titles, so the PR title must carry
  the conventional ``!`` breaking marker.
"""

from __future__ import annotations

import ast
import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

from tests.test_sidekick_public_api_stability import (
    extract_class_info,
    extract_signature_from_function,
    get_ast_node_for_symbol,
)

log = logging.getLogger(__name__)

pytestmark = [pytest.mark.unit, pytest.mark.contract, pytest.mark.headless_safe]

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_ROOT = REPO_ROOT / "src" / "shared" / "python"
BASELINE_DIR = REPO_ROOT / "tests" / "api_baselines"

# Package name -> path relative to src/shared/python (directory or module).
VENDORED_PACKAGES: dict[str, str] = {
    "theme": "theme",
    "plot_theme": "plot_theme",
    "golf_club": "golf_club",
    "swing_sim": "swing_sim",
    "launch_monitor": "launch_monitor",
    "contracts": "contracts.py",
    "safe_eval": "safe_eval.py",
}


def _package_modules(package: str) -> list[Path]:
    target = SHARED_ROOT / VENDORED_PACKAGES[package]
    if target.is_file():
        return [target]
    modules: list[Path] = []
    for root, dirs, files in os.walk(target):
        dirs[:] = sorted(d for d in dirs if d not in {"tests", "__pycache__"})
        modules.extend(Path(root) / f for f in sorted(files) if f.endswith(".py"))
    return modules


def _public_names(module_ast: ast.Module) -> tuple[list[str], bool]:
    """Return (names, explicit) where explicit means ``__all__`` was declared."""
    for node in module_ast.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        return (
                            [ast.unparse(elt).strip("'\"") for elt in node.value.elts],
                            True,
                        )
    names: list[str] = []
    for node in module_ast.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    names.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and not node.target.id.startswith("_"):
                names.append(node.target.id)
    return sorted(dict.fromkeys(names)), False


def extract_public_api(path: Path) -> dict[str, Any]:
    """AST-extract the public surface of one module (no import side effects)."""
    module_ast = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names, explicit = _public_names(module_ast)
    symbols: dict[str, Any] = {}
    for symbol in names:
        found = get_ast_node_for_symbol(module_ast, symbol, path)
        if found is None:
            symbols[symbol] = {"type": "reexport"}
            continue
        node, resolved = found
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols[symbol] = {
                "type": "function",
                "signature": extract_signature_from_function(node),
            }
        elif isinstance(node, ast.ClassDef):
            symbols[symbol] = {
                "type": "class",
                "info": extract_class_info(node, resolved),
            }
        else:
            symbols[symbol] = {"type": "variable"}
    return {"__all__": names, "explicit_all": explicit, "symbols": symbols}


def snapshot_package(package: str) -> dict[str, Any]:
    target = SHARED_ROOT / VENDORED_PACKAGES[package]
    base = target.parent if target.is_file() else target
    return {
        (path.relative_to(base).as_posix()): extract_public_api(path)
        for path in _package_modules(package)
    }


def baseline_path(package: str) -> Path:
    return BASELINE_DIR / f"{package}_api_baseline.json"


def _diff(baseline: dict[str, Any], current: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for module in sorted(set(baseline) - set(current)):
        problems.append(f"{module}: module removed")
    for module, expected in baseline.items():
        actual = current.get(module)
        if actual is None:
            continue
        missing = [n for n in expected["__all__"] if n not in actual["__all__"]]
        for name in missing:
            problems.append(f"{module}: public symbol {name!r} removed")
        for name, sym in expected["symbols"].items():
            if name in actual["symbols"] and actual["symbols"][name] != sym:
                problems.append(
                    f"{module} / {name}: signature changed\n"
                    f"  expected: {json.dumps(sym, sort_keys=True)}\n"
                    f"  got:      {json.dumps(actual['symbols'][name], sort_keys=True)}"
                )
    return problems


@pytest.mark.parametrize("package", sorted(VENDORED_PACKAGES))
def test_vendored_package_public_api_matches_baseline(
    package: str, pytestconfig: pytest.Config
) -> None:
    current = snapshot_package(package)
    path = baseline_path(package)

    if pytestconfig.getoption("--regenerate-api-baseline"):
        BASELINE_DIR.mkdir(exist_ok=True)
        path.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        log.info("Regenerated %s", path)
        return

    assert path.is_file(), (
        f"{path} missing; run pytest {__file__} --regenerate-api-baseline"
    )
    baseline = json.loads(path.read_text(encoding="utf-8"))
    problems = _diff(baseline, current)
    if problems:
        pytest.fail(
            f"Public API of shared.python.{package} changed incompatibly. Additions are "
            "fine; removals and signature changes need a baseline bump in the same PR "
            "plus downstream issues (see module docstring):\n" + "\n".join(problems)
        )
    # Additions are allowed but must be recorded so the baseline stays honest.
    added = sorted(set(current) - set(baseline))
    new_symbols = {
        module: sorted(
            set(current[module]["__all__"]) - set(baseline[module]["__all__"])
        )
        for module in baseline
        if module in current
        and set(current[module]["__all__"]) - set(baseline[module]["__all__"])
    }
    assert not added and not new_symbols, (
        f"shared.python.{package} gained public surface not in the baseline "
        f"(new modules {added}, new symbols {new_symbols}); regenerate the baseline "
        "in this PR so the export is reviewed"
    )


def test_every_vendored_package_has_a_baseline() -> None:
    for package in VENDORED_PACKAGES:
        assert baseline_path(package).is_file(), package
        assert (SHARED_ROOT / VENDORED_PACKAGES[package]).exists(), package


def test_baselines_are_not_empty() -> None:
    """A baseline with no public symbols would guard nothing."""
    for package in VENDORED_PACKAGES:
        data = json.loads(baseline_path(package).read_text(encoding="utf-8"))
        total = sum(len(module["__all__"]) for module in data.values())
        assert total > 0, package
