"""Require matched visual evidence when Rate-of-Closure tab surfaces change."""

from __future__ import annotations

import argparse
import fnmatch
import logging
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

LOGGER = logging.getLogger(__name__)

SHARED_MANIFEST: Final = "src/rate_of_closure/visualization_tabs.v1.json"
SHARED_AUDIT: Final = "docs/audits/rate_of_closure_visual_first_epic_4433.v1.json"
REACT_FIRST_VIEWPORT_TEST: Final = (
    "src/rate_of_closure/web/e2e/visualization-tab-visibility.spec.ts"
)
PYQT_FIRST_VIEWPORT_TEST: Final = (
    "tests/rate_of_closure/test_pyqt_visualization_tab_visibility.py"
)

_REACT_PATTERNS: Final = (
    "src/rate_of_closure/web/src/components/*.tsx",
    "src/rate_of_closure/web/src/components/**/*.tsx",
    "src/rate_of_closure/web/src/App.tsx",
    "src/rate_of_closure/web/src/*.css",
    "src/rate_of_closure/web/src/**/*.css",
)
_PYQT_PATTERNS: Final = (
    "src/rate_of_closure/ui/pyqt6/*tab*.py",
    "src/rate_of_closure/ui/pyqt6/*visual*.py",
    "src/rate_of_closure/ui/pyqt6/main_window*.py",
    "src/rate_of_closure/ui/pyqt6/app_style.py",
)
_SHARED_VISUAL_PATHS: Final = frozenset(
    {
        "src/rate_of_closure/plot_workspace_limits.py",
        "src/rate_of_closure/visual_layout_preferences.py",
    }
)


def _normalize_paths(paths: Iterable[str]) -> frozenset[str]:
    """Return nonempty repository-relative POSIX paths."""

    normalized = {path.strip().replace("\\", "/").removeprefix("./") for path in paths}
    return frozenset(path for path in normalized if path)


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    """Return whether ``path`` matches one declared material-visual pattern."""

    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _surface_requirements(surface: str) -> tuple[str, ...]:
    """Return the exact evidence co-change contract for one visual surface."""

    if surface == "react":
        return SHARED_MANIFEST, SHARED_AUDIT, REACT_FIRST_VIEWPORT_TEST
    if surface == "pyqt":
        return SHARED_MANIFEST, SHARED_AUDIT, PYQT_FIRST_VIEWPORT_TEST
    raise ValueError(f"unknown visual surface: {surface}")


def validate_visual_evidence_changes(changed_files: Iterable[str]) -> tuple[str, ...]:
    """Return deterministic errors for incomplete visual-evidence co-changes.

    Preconditions:
        Paths are repository-relative strings. Both slash conventions are
        accepted.
    Postconditions:
        An empty result means every triggered surface includes its manifest,
        audit, and first-viewport evidence update in the same change set.
    """

    changed = _normalize_paths(changed_files)
    surfaces: list[str] = []
    if any(_matches_any(path, _REACT_PATTERNS) for path in changed):
        surfaces.append("react")
    if any(_matches_any(path, _PYQT_PATTERNS) for path in changed):
        surfaces.append("pyqt")
    if changed.intersection(_SHARED_VISUAL_PATHS):
        surfaces = ["react", "pyqt"]

    errors: list[str] = []
    for surface in surfaces:
        for required in _surface_requirements(surface):
            if required not in changed:
                errors.append(f"{surface} visual changes require {required}")
    return tuple(errors)


def _git_changed_files(base_ref: str) -> tuple[str, ...]:
    """Read changed paths from an exact merge-base comparison or fail closed."""

    if not base_ref.strip():
        raise ValueError("base_ref must be nonempty")
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMRTUXB", f"{base_ref}...HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return tuple(result.stdout.splitlines())


def _file_changed_paths(path: Path) -> tuple[str, ...]:
    """Read a deterministic newline-delimited changed-path fixture."""

    if not path.is_file():
        raise FileNotFoundError(f"changed-files input does not exist: {path}")
    return tuple(path.read_text(encoding="utf-8").splitlines())


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-ref", help="Git base revision for base...HEAD")
    source.add_argument("--changed-files", type=Path, help="Newline-delimited paths")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the changed-path contract and return a process exit code."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)
    try:
        paths = (
            _git_changed_files(args.base_ref)
            if args.base_ref is not None
            else _file_changed_paths(args.changed_files)
        )
        errors = validate_visual_evidence_changes(paths)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        LOGGER.error("visual evidence governance could not evaluate changes: %s", exc)
        return 2
    for error in errors:
        LOGGER.error(error)
    if errors:
        return 1
    LOGGER.info("Rate-of-Closure visual evidence co-change contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
