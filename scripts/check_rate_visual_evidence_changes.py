"""Require matched visual evidence when Rate-of-Closure tab surfaces change."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import logging
import os
import re
import subprocess
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Final

LOGGER = logging.getLogger(__name__)

SHARED_MANIFEST: Final = "src/rate_of_closure/visualization_tabs.v1.json"
ACCEPTANCE_MANIFEST: Final = "src/rate_of_closure/visualization_acceptance.v1.json"
SHARED_AUDIT: Final = "docs/audits/rate_of_closure_visual_first_epic_4433.v1.json"
REACT_FIRST_VIEWPORT_TEST: Final = (
    "src/rate_of_closure/web/e2e/visualization-tab-visibility.spec.ts"
)
PYQT_FIRST_VIEWPORT_TEST: Final = (
    "tests/rate_of_closure/test_pyqt_visualization_tab_visibility.py"
)

_EXEMPTION_PATTERN: Final = re.compile(
    r"(?mi)^\s*(?:rate[-_]visual[-_]exempt(?:ion)?|no[-_]visual[-_]change)\s*(?::\s*(.+))?$"
)

_REACT_PATTERNS: Final = (
    "src/rate_of_closure/web/src/components/*.tsx",
    "src/rate_of_closure/web/src/components/**/*.tsx",
    "src/rate_of_closure/web/src/App.tsx",
    "src/rate_of_closure/web/src/*.css",
    "src/rate_of_closure/web/src/**/*.css",
)
_PYQT_PATTERNS: Final = (
    "src/rate_of_closure/ui/pyqt6/*.ui",
    "src/rate_of_closure/ui/pyqt6/**/*.ui",
    "src/rate_of_closure/ui/pyqt6/*tab*.py",
    "src/rate_of_closure/ui/pyqt6/*visual*.py",
    "src/rate_of_closure/ui/pyqt6/main_window*.py",
    "src/rate_of_closure/ui/pyqt6/app_style.py",
    "src/rate_of_closure/ui/pyqt6/*scene*.py",
    "src/rate_of_closure/ui/pyqt6/*canvas*.py",
    "src/rate_of_closure/ui/pyqt6/*paint*.py",
    "src/rate_of_closure/ui/pyqt6/*render*.py",
)
_PAINT_SYMBOLS_PATTERN: Final = re.compile(
    r"\b(paintEvent|QPainter|QPaintEvent|drawBackground|drawForeground|QPainterPath|drawPath|drawRect)\b"
)


def _normalize_paths(paths: Iterable[str]) -> frozenset[str]:
    """Return nonempty repository-relative POSIX paths."""

    normalized = {path.strip().replace("\\", "/").removeprefix("./") for path in paths}
    return frozenset(path for path in normalized if path)


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    """Return whether ``path`` matches one declared material-visual pattern."""

    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _is_react_surface_path(path: str) -> bool:
    """Exclude test modules from the shipped React visual surface."""

    return _matches_any(path, _REACT_PATTERNS) and not path.endswith(
        (".test.tsx", ".spec.tsx")
    )


def _is_pyqt_surface_path(path: str) -> bool:
    """Return whether path matches a shipped PyQt visual or paint surface."""

    if not path.startswith("src/rate_of_closure/ui/pyqt6/"):
        return False
    if path.endswith((".test.py", "_test.py")):
        return False
    if _matches_any(path, _PYQT_PATTERNS):
        return True
    file_path = Path(path)
    if file_path.is_file():
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            return _PAINT_SYMBOLS_PATTERN.search(content) is not None
        except OSError:
            pass
    return False


def _surface_requirements(surface: str) -> tuple[str, ...]:
    """Return the exact evidence co-change contract for one visual surface."""

    if surface == "react":
        return (
            SHARED_MANIFEST,
            ACCEPTANCE_MANIFEST,
            SHARED_AUDIT,
            REACT_FIRST_VIEWPORT_TEST,
        )
    if surface == "pyqt":
        return (
            SHARED_MANIFEST,
            ACCEPTANCE_MANIFEST,
            SHARED_AUDIT,
            PYQT_FIRST_VIEWPORT_TEST,
        )
    raise ValueError(f"unknown visual surface: {surface}")


def is_substantive_evidence_change(
    path: str, base_content: str, current_content: str
) -> bool:
    """Return whether current_content has a changed substantive content hash.

    For JSON evidence, canonicalizes structure so whitespace-only or trailing
    newline edits do not satisfy the gate. For code/test fixtures, normalizes
    by stripping empty lines and trailing whitespace per line.
    """

    if path.endswith(".json"):
        try:
            base_obj = json.loads(base_content)
            curr_obj = json.loads(current_content)
            base_canonical = json.dumps(base_obj, sort_keys=True, separators=(",", ":"))
            curr_canonical = json.dumps(curr_obj, sort_keys=True, separators=(",", ":"))
            return (
                hashlib.sha256(base_canonical.encode("utf-8")).digest()
                != hashlib.sha256(curr_canonical.encode("utf-8")).digest()
            )
        except (ValueError, TypeError):
            pass
    base_norm = "\n".join(
        line.rstrip() for line in base_content.splitlines() if line.strip()
    )
    curr_norm = "\n".join(
        line.rstrip() for line in current_content.splitlines() if line.strip()
    )
    return (
        hashlib.sha256(base_norm.encode("utf-8")).digest()
        != hashlib.sha256(curr_norm.encode("utf-8")).digest()
    )


def _git_show_blob(base_ref: str, path: str) -> str | None:
    """Read a path at base_ref via git show, returning None if absent."""

    try:
        result = subprocess.run(
            ["git", "show", f"{base_ref}:{path}"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _file_content(path: str) -> str | None:
    """Read current content of a repository file, returning None if unreadable."""

    file_path = Path(path)
    if file_path.is_file():
        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
    return None


def extract_exemption_reason(text: str) -> str | None:
    """Extract an explicit visual-evidence exemption reason from text."""

    if not text:
        return None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = _EXEMPTION_PATTERN.match(line)
        if match:
            reason = (match.group(1) or "").strip()
            return reason or "declared no-visual-change marker"
    return None


def validate_visual_evidence_changes(
    changed_files: Iterable[str],
    *,
    exemption_reason: str | None = None,
    base_ref: str | None = None,
    evidence_content_resolver: Callable[[str], tuple[str, str] | None] | None = None,
) -> tuple[str, ...]:
    """Return deterministic errors for incomplete visual-evidence co-changes.

    Preconditions:
        Paths are repository-relative strings. Both slash conventions are
        accepted.
    Postconditions:
        An empty result means every triggered surface includes its manifest,
        acceptance authority, audit, and first-viewport evidence update in the
        same change set with a verified substantive content hash change, or an
        explicit non-empty exemption reason was declared. If the change set
        touches no .tsx, .css, .ui, or paint code, it skips unconditionally.
    """

    if exemption_reason is not None and not exemption_reason.strip():
        raise ValueError("exemption_reason must be nonempty when provided")

    changed = _normalize_paths(changed_files)
    surfaces: list[str] = []
    if any(_is_react_surface_path(path) for path in changed):
        surfaces.append("react")
    if any(_is_pyqt_surface_path(path) for path in changed):
        surfaces.append("pyqt")

    if not surfaces:
        return ()

    if exemption_reason is not None:
        LOGGER.info(
            "Rate-of-Closure visual evidence requirements exempted: %s",
            exemption_reason.strip(),
        )
        return ()

    errors: list[str] = []
    for surface in surfaces:
        for required in _surface_requirements(surface):
            if required not in changed:
                errors.append(f"{surface} visual changes require {required}")
            elif evidence_content_resolver is not None:
                resolved = evidence_content_resolver(required)
                if resolved is not None:
                    base_text, curr_text = resolved
                    if not is_substantive_evidence_change(
                        required, base_text, curr_text
                    ):
                        errors.append(
                            f"{surface} visual changes require substantive update to {required} "
                            "(content hash unchanged from base)"
                        )
            elif base_ref is not None:
                git_base = _git_show_blob(base_ref, required)
                disk_curr = _file_content(required)
                if git_base is not None and disk_curr is not None:
                    if not is_substantive_evidence_change(
                        required, git_base, disk_curr
                    ):
                        errors.append(
                            f"{surface} visual changes require substantive update to {required} "
                            "(content hash unchanged from base)"
                        )
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


def _git_commit_messages(base_ref: str) -> str:
    """Read commit messages between base_ref and HEAD."""

    if not base_ref.strip():
        raise ValueError("base_ref must be nonempty")
    result = subprocess.run(
        ["git", "log", "--format=%B", f"{base_ref}...HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def _env_exemption_reason() -> str | None:
    """Check environment variables for a declared visual exemption."""

    for var_name in ("RATE_VISUAL_EXEMPTION", "PR_BODY"):
        value = os.environ.get(var_name)
        if value and value.strip():
            extracted = extract_exemption_reason(value)
            if extracted:
                return extracted
            if var_name == "RATE_VISUAL_EXEMPTION":
                return value.strip()
    return None


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
    parser.add_argument(
        "--exemption-reason",
        help="Explicit declared reason why visual evidence co-change is not required",
    )
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
        exemption = args.exemption_reason
        if exemption is None and args.base_ref is not None:
            try:
                commit_msgs = _git_commit_messages(args.base_ref)
                exemption = extract_exemption_reason(commit_msgs)
            except (OSError, subprocess.CalledProcessError):
                pass
        if exemption is None:
            exemption = _env_exemption_reason()
        errors = validate_visual_evidence_changes(
            paths, exemption_reason=exemption, base_ref=args.base_ref
        )
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
