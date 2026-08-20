"""Every first-party import in ``src/`` must name a module that actually exists.

Nothing else in the toolchain enforces this. ``mypy.ini`` sets
``ignore_missing_imports = True`` (and CI's delta-mypy passes
``--ignore-missing-imports`` on top), which is the *sole* reason
``import-not-found`` never fires — ``--follow-imports=skip`` does not suppress
it. Ruff never resolves imports, so a deferred ``from x.y import z`` where
``x.y`` has never existed is invisible to F401 because ``z`` *is* used.

The failure mode this catches is silent, not loud. Three live examples found
when this guard was written:

- ``StandalonePreferences.apply_tokens()`` imported ``theme.sidekick_tokens``,
  a module that existed nowhere in the repo. Calling it raised
  ``ModuleNotFoundError``; it had no callers and no tests, so nothing noticed.
- ``signal_toolkit.polynomial_generator`` imported ``shared.python
  .logging_config`` (real path: ``shared.python.logging_pkg.logging_config``)
  inside ``try/except ImportError``, so it silently fell back to bare logging
  forever.
- ``model_generation.humanoid`` wrapped eight imports in one
  ``try/except ImportError: pass``. One missing module aborted the whole block,
  so all 34 names in its ``__all__`` were dead — and ``# mypy: ignore-errors``
  at the top of the file guaranteed nobody would find out.

Resolution is a **filesystem path walk**, deliberately not
``importlib.util.find_spec``: find_spec reports failure when a *parent* package
raises on import (a false positive — the module is right there), it depends on
the ambient ``sys.path`` rather than the roots this repo declares, and it
executes package ``__init__`` side effects during a static check.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

# Vendored self-contained sub-apps run their suites in their origin repos and
# are excluded from ruff/mypy/bandit here too — mirror those exclude lists.
_EXCLUDED_TREES = frozenset({"movement_optimizer", "pendulum_simulator"})

# Modules a *downstream host* supplies at runtime; absent from this repo by
# design, not by mistake. Entries must name the providing repo and be reachable
# only from code that already tolerates their absence.
#
# ``shared.python.launcher_embed`` is UpstreamDrift's embeddable-tool contract
# (``src/shared/python/launcher_embed/`` there). ``data_explorer`` was migrated
# into Tools carrying its embed adapter; the adapter registers itself when
# UpstreamDrift's launcher hosts it and is skipped otherwise, which is why its
# import sits inside ``contextlib.suppress(ImportError)``.
_HOST_PROVIDED = frozenset({"shared.python.launcher_embed"})

# Ratchet for genuine debt. This set is EMPTY and adding to it is not a routine
# fix: an entry means production code imports a module that does not exist, so
# that code path is dead. If you must add one, link a tracked issue in a comment
# beside it. Removing entries needs no ceremony.
_KNOWN_UNRESOLVED: frozenset[str] = frozenset()


@functools.lru_cache(maxsize=1)
def _import_roots() -> tuple[Path, ...]:
    """Return every directory that acts as a package-discovery root for ``src/``.

    Cached: ``_resolves`` is called once per import node across the whole tree,
    and rebuilding this per call is what made a sibling guard take 67 s.
    """
    roots = [
        REPO_ROOT,  # ``src.shared.python.x`` spellings
        SRC_ROOT,  # ``shared.python.x`` and top-level tool packages
        SRC_ROOT / "shared" / "python",  # bare ``sidekick.x`` legacy spellings
        SRC_ROOT / "python" / "src",
    ]
    # Sub-app layouts: src/<tool>/python/<tool>/... and src/<a>/<b>/python/<b>/...
    roots.extend(sorted(SRC_ROOT.glob("*/python")))
    roots.extend(sorted(SRC_ROOT.glob("*/*/python")))
    return tuple(dict.fromkeys(r for r in roots if r.is_dir()))


@functools.lru_cache(maxsize=1)
def _first_party_names() -> frozenset[str]:
    """Top-level names importable from a declared root (i.e. not third-party)."""
    names: set[str] = set()
    for root in _import_roots():
        for child in root.iterdir():
            if child.is_dir() and not child.name.startswith((".", "__")):
                names.add(child.name)
            elif child.suffix == ".py" and child.stem != "__init__":
                names.add(child.stem)
    return frozenset(names)


@functools.cache
def _resolves(module: str) -> bool:
    parts = module.split(".")
    for root in _import_roots():
        candidate = root.joinpath(*parts)
        if candidate.with_suffix(".py").is_file():
            return True
        if (candidate / "__init__.py").is_file():
            return True
        if candidate.is_dir():  # PEP 420 namespace package
            return True
    return False


def _is_production_file(path: Path) -> bool:
    """Production code only — mirrors the #3316 guard's tests exemption.

    Test files are out of scope: several import ``tools.folder_tools.*``
    submodules that do not exist, and that tree is already excluded from
    ruff/mypy/bandit in ci-standard.yml as known pre-existing debt.
    """
    relative = path.relative_to(REPO_ROOT)
    if "__pycache__" in relative.parts:
        return False
    if len(relative.parts) > 1 and relative.parts[1] in _EXCLUDED_TREES:
        return False
    if "tests" in relative.parts or path.name.startswith("test_"):
        return False
    return True


def _imported_modules(tree: ast.AST) -> list[tuple[int, str]]:
    """Absolute import targets in *tree* as ``(lineno, dotted_module)`` pairs.

    Relative imports are skipped: the interpreter resolves them against the
    containing package, so they cannot name a nonexistent tree the way an
    absolute spelling can.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
    return found


def _unresolvable_imports() -> list[str]:
    violations: list[str] = []
    exempt = _HOST_PROVIDED | _KNOWN_UNRESOLVED
    first_party = _first_party_names()

    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        if not _is_production_file(py_file):
            continue
        try:
            tree = ast.parse(
                py_file.read_text(encoding="utf-8", errors="replace"),
                filename=str(py_file),
            )
        except SyntaxError:
            continue

        for lineno, module in _imported_modules(tree):
            if module in exempt or module.split(".")[0] not in first_party:
                continue
            if not _resolves(module):
                relative = py_file.relative_to(REPO_ROOT).as_posix()
                violations.append(f"{relative}:{lineno}: {module}")
    return violations


def test_production_first_party_imports_resolve() -> None:
    """No production module may import a first-party module that does not exist."""
    violations = _unresolvable_imports()
    assert not violations, (
        "Production code imports first-party modules that do not exist. "
        "Each one is a dead code path — the import either raises at runtime or "
        "is swallowed by an except/suppress block that hides the breakage:\n  "
        + "\n  ".join(violations)
    )


def test_host_provided_allowlist_is_still_needed() -> None:
    """Drop allowlist entries once the module exists locally.

    Without this, an entry added for a genuinely absent module silently becomes
    a permanent exemption for one that is now present and checkable.
    """
    now_resolvable = sorted(
        m for m in _HOST_PROVIDED | _KNOWN_UNRESOLVED if _resolves(m)
    )
    assert not now_resolvable, (
        "These modules now resolve in-repo and must be removed from "
        f"_HOST_PROVIDED / _KNOWN_UNRESOLVED: {now_resolvable}"
    )


def test_allowlisted_modules_are_actually_imported() -> None:
    """An exemption for a module nobody imports is dead configuration."""
    imported: set[str] = set()
    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        if not _is_production_file(py_file):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        imported.update(module for _, module in _imported_modules(tree))

    unused = sorted((_HOST_PROVIDED | _KNOWN_UNRESOLVED) - imported)
    assert not unused, (
        f"Allowlist entries no longer imported by any production file: {unused}"
    )
