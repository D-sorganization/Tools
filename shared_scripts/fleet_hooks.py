#!/usr/bin/env python3
"""Portable fleet hook checks used by Repository_Management templates.

The checks in this module are intentionally fast and deterministic. They are
meant to catch failures that should never reach CI, while leaving matrix,
container, integration, and deployment checks to GitHub Actions.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType

logger = logging.getLogger(__name__)

ROOT = Path.cwd()

# Root entries exempt from the scratch-suffix check in check_root_clutter.
ROOT_ALLOWLIST = {
    ".gitignore",
    ".pre-commit-config.yaml",
    "AGENTS.md",
    "CLAUDE.md",
    "LICENSE",
    "Makefile",
    "README.md",
    "SPEC.md",
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
    # Legitimate root entries in any Rust-bearing repo (issue #4486). These are
    # inert under the current deny-scratch default — none of them carries a
    # scratch suffix — and exist so the allowlist is already correct if the
    # stricter deny-unless-allowlisted default is ever adopted.
    "Cargo.toml",
    "Cargo.lock",
    "target",
}
ROOT_SCRATCH_SUFFIXES = {".log", ".tmp", ".bak", ".zip", ".7z"}

DEFAULT_MAX_BYTES = 1_000_000
DEFAULT_MAX_SOURCE_LINES = 1500
OVERSIZED_SOURCE_LINE_BASELINES = {
    # Legacy MATLAB GUI monolith tracked by Tools #3359. It may be touched for
    # cleanup, but it must not grow beyond this frozen line-count budget.
    (
        "src/media_processing/audio_processor/matlab/audio_signal_processor/gui/"
        "MainWindow.m"
    ): 3528,
    # Legacy Modern Robotics compatibility module tracked in
    # scripts/monolith_baseline.txt. It may be touched for contract cleanup,
    # but cannot grow beyond this frozen budget without a split.
    "src/rotation_converter/modern_robotics.py": 2130,
}

SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".js",
    ".jsx",
    ".m",
    ".py",
    ".rs",
    ".ts",
    ".tsx",
}
SOURCE_PREFIXES = ("src/", "app/", "backend/", "frontend/", "scripts/")
DOC_GOVERNANCE_PREFIXES = ("docs/adr/", "docs/architecture/adr/")
DEPENDENCY_MANIFESTS = (
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "poetry.lock",
    "uv.lock",
    "Pipfile.lock",
)
BANNED_WORKFLOW_TOKENS = ("d-sorg-fleet-14core",)
ALLOWED_WORKFLOW_TOKENS = {
    ("Bot-CI-Trigger.yml", "d-sorg-fleet-14core"),
    ("heavy-integration-tests.yml", "d-sorg-fleet-14core"),
    ("heavy-tests-opt-in.yml", "d-sorg-fleet-14core"),
}
BARE_EXCEPT_RE = re.compile(r"^\s*except\s*:\s*(?:#.*)?$")


def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    # Decode explicitly as UTF-8 with replacement rather than inheriting the
    # locale codec. On Windows that default is cp1252, which raises
    # UnicodeDecodeError partway through any `git show`/`git diff` output
    # carrying a character it cannot map -- and SPEC.md is 3 MB of em dashes,
    # arrows and box characters. Matches Repository_Management's copy.
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _git_files(args: Sequence[str]) -> list[str]:
    result = _run(["git", *args])
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def staged_files() -> list[str]:
    return _git_files(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])


def changed_files() -> list[str]:
    """Return files changed in the current event scope.

    Scope, in priority order:

    1. Staged (``git diff --cached``) — pre-commit.
    2. Working-tree diff vs ``HEAD`` (``git diff HEAD``) — pre-commit when
       the user did ``git add -p`` and left some changes unstaged.
    3. ``@{upstream}..HEAD`` — pre-push when the tracking ref is set.
    4. ``origin/main...HEAD`` — pre-push fallback when no upstream is set.

    Returns an empty list when no scope is detectable (clean tree, no
    upstream, etc.). Callers must treat empty as "nothing to check".

    The historical ``ls-files`` fallback that scanned the entire working
    tree was the leading cause of agents reaching for ``--no-verify``:
    it surfaced pre-existing grandfathered violations on every clean
    commit. It is intentionally removed — if your diff is empty, the
    hook is a no-op.
    """
    files = set(staged_files())
    files.update(_git_files(["diff", "--name-only", "--diff-filter=ACMR", "HEAD"]))
    if not files:
        files.update(
            _git_files(
                ["diff", "--name-only", "--diff-filter=ACMR", "@{upstream}..HEAD"]
            )
        )
    if not files:
        files.update(
            _git_files(
                ["diff", "--name-only", "--diff-filter=ACMR", "origin/main...HEAD"]
            )
        )
    return sorted(files)


def existing(paths: Iterable[str]) -> list[Path]:
    return [ROOT / path for path in paths if (ROOT / path).is_file()]


def rel(path: Path) -> str:
    return path.as_posix()


def repo_rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def is_source(path: str) -> bool:
    posix = path.replace("\\", "/")
    suffix = Path(posix).suffix.lower()
    return suffix in SOURCE_SUFFIXES and (
        posix.startswith(SOURCE_PREFIXES)
        or posix.startswith("tests/")
        or posix.startswith(".github/workflows/")
    )


def fail_or_warn(title: str, failures: list[str], warn_only: bool) -> int:
    if not failures:
        return 0
    label = "WARNING" if warn_only else "ERROR"
    log_fn = logger.warning if warn_only else logger.error
    log_fn("%s: %s", label, title)
    for failure in failures:
        log_fn("  - %s", failure)
    return 0 if warn_only else 1


def check_file_size(args: argparse.Namespace) -> int:
    failures: list[str] = []
    for path in existing(changed_files()):
        if ".git/" in rel(path) or "node_modules/" in rel(path):
            continue
        size = path.stat().st_size
        if size > args.max_bytes:
            failures.append(f"{rel(path)} is {size} bytes; limit is {args.max_bytes}")
            continue
        if path.suffix.lower() in SOURCE_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                lines = text.count("\n") + 1
            except OSError:
                continue
            oversized_baseline = OVERSIZED_SOURCE_LINE_BASELINES.get(repo_rel(path))
            if oversized_baseline is not None:
                if lines > oversized_baseline:
                    failures.append(
                        f"{rel(path)} has {lines} lines; baseline is "
                        f"{oversized_baseline}"
                    )
                continue
            if lines > args.max_source_lines:
                failures.append(
                    f"{rel(path)} has {lines} lines; limit is {args.max_source_lines}"
                )
    return fail_or_warn("file size budget exceeded", failures, args.warn_only)


def check_spec_freshness(args: argparse.Namespace) -> int:
    spec = ROOT / "SPEC.md"
    if not spec.exists():
        return 0
    files = staged_files() or changed_files()
    source_changes = [path for path in files if is_source(path)]
    if not source_changes or "SPEC.md" in files:
        return 0
    sample = ", ".join(source_changes[:5])
    failures = [f"source/config changed without staging SPEC.md: {sample}"]
    return fail_or_warn("SPEC.md freshness boundary", failures, args.warn_only)


def _load_spec_changelog() -> ModuleType | None:
    """Load the portable change-log helper, or ``None`` if it was not shipped."""
    module_path = Path(__file__).with_name("spec_changelog.py")
    if not module_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("fleet_spec_changelog", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        return None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module defines dataclasses, and
    # dataclasses.field resolution looks the defining module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _spec_text_at(ref: str) -> str | None:
    result = _run(["git", "show", f"{ref}:SPEC.md"])
    return result.stdout if result.returncode == 0 else None


def check_spec_changelog(args: argparse.Namespace) -> int:
    """Enforce PR-keyed SPEC.md change-log rows (Repository_Management#1520).

    Rows are ``| YYYY-MM-DD | #<pr> | summary |``. There is no serial version
    to collide on and no header field to keep in lockstep, so two concurrent
    pull requests cannot conflict by construction. What is still enforced:

    * the table parses and every row is well formed;
    * a PR key is never reused (a second row for the same pull request means
      somebody copied a row instead of editing their own);
    * a change that touches source/config **and** SPEC.md actually adds a row,
      which is the requirement the old gate existed to impose.
    """
    spec = ROOT / "SPEC.md"
    if not spec.exists():
        return 0
    module = _load_spec_changelog()
    if module is None:
        return fail_or_warn(
            "SPEC.md change-log checker missing",
            ["shared_scripts/spec_changelog.py was not shipped to this repository"],
            True,
        )

    failures: list[str] = []
    after = spec.read_text(encoding="utf-8")
    try:
        changelog = module.parse_changelog(after)
    except module.SpecChangelogError as exc:
        return fail_or_warn(
            "SPEC.md change-log boundary",
            [f"SPEC.md: {exc}"],
            args.warn_only,
        )

    failures.extend(module.validate(changelog))

    files = staged_files() or changed_files()
    if "SPEC.md" in files and any(is_source(path) for path in files):
        base_ref = os.environ.get("FLEET_HOOK_FROM_REF") or "HEAD"
        before = _spec_text_at(base_ref)
        if before is not None and not module.rows_added(before, after):
            failures.append(
                "SPEC.md was edited alongside source/config changes but no "
                "change-log row was added. Add one row for your pull request: "
                "| YYYY-MM-DD | #<pr> | summary |"
            )

    return fail_or_warn("SPEC.md change-log boundary", failures, args.warn_only)


def check_adr_readme(args: argparse.Namespace) -> int:
    files = staged_files() or changed_files()
    adr_changes = [
        path
        for path in files
        if path.endswith(".md")
        and any(
            path.replace("\\", "/").startswith(prefix)
            for prefix in DOC_GOVERNANCE_PREFIXES
        )
        and not path.endswith("README.md")
    ]
    if not adr_changes:
        return 0
    readmes = [prefix + "README.md" for prefix in DOC_GOVERNANCE_PREFIXES]
    if any((ROOT / readme).exists() and readme in files for readme in readmes):
        return 0
    failures = [f"ADR changed without staging ADR README: {', '.join(adr_changes[:5])}"]
    return fail_or_warn("ADR README sync boundary", failures, args.warn_only)


def check_workflow_inventory(args: argparse.Namespace) -> int:
    workflow_dir = ROOT / ".github" / "workflows"
    if not workflow_dir.exists():
        return 0
    failures: list[str] = []
    for path in sorted(workflow_dir.glob("*.y*ml")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for token in BANNED_WORKFLOW_TOKENS:
                if token in line and (path.name, token) not in ALLOWED_WORKFLOW_TOKENS:
                    failures.append(f"{rel(path)}:{line_no} contains {token!r}")
    return fail_or_warn("workflow inventory drift", failures, args.warn_only)


def check_error_handling(args: argparse.Namespace) -> int:
    failures: list[str] = []
    for path in existing(changed_files()):
        if path.suffix.lower() != ".py":
            continue
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
        ):
            if BARE_EXCEPT_RE.match(line):
                failures.append(f"{rel(path)}:{line_no} uses bare except")
    return fail_or_warn("error-handling ratchet failed", failures, args.warn_only)


def check_root_clutter(args: argparse.Namespace) -> int:
    """Reject scratch/output files staged at the repo root.

    Note the direction of the test: this is *deny-scratch*, not
    *deny-unless-allowlisted*. ``ROOT_ALLOWLIST`` therefore only exempts an
    entry from the suffix check below — it is not the set of things permitted
    at the root. Inverting that default would newly reject 30-40 tracked,
    ordinary root files (``.gitattributes``, ``.gitmodules``, ``.editorconfig``,
    ...) in every fleet repo; see issue #4486 for the measurements and for the
    ``--warn-only`` rollout that any such change needs.
    """
    failures: list[str] = []
    for path in staged_files():
        posix = path.replace("\\", "/")
        if "/" in posix or posix in ROOT_ALLOWLIST:
            continue
        if Path(posix).suffix.lower() in ROOT_SCRATCH_SUFFIXES:
            failures.append(f"{posix} looks like root-level scratch/output")
    return fail_or_warn("repo root clutter", failures, args.warn_only)


def check_dependency_direction(args: argparse.Namespace) -> int:
    """Catch obvious shared/provider direction mistakes.

    This is intentionally light. Repo-specific architecture tests should own the
    full dependency graph.
    """
    failures: list[str] = []
    for path in existing(changed_files()):
        posix = rel(path)
        if "src/shared/" not in posix or path.suffix.lower() != ".py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        banned = (
            "from tools",
            "import tools",
            "from gasification",
            "from upstream_drift",
        )
        for token in banned:
            if token in text:
                failures.append(
                    f"{posix} appears to import downstream package {token!r}"
                )
    return fail_or_warn("dependency direction ratchet failed", failures, args.warn_only)


def check_dependency_audit(args: argparse.Namespace) -> int:
    if shutil.which("pip-audit") is None:
        return fail_or_warn(
            "pip-audit unavailable",
            ["install pip-audit or run through the pre-commit hook environment"],
            args.warn_only,
        )
    manifests = [path for path in DEPENDENCY_MANIFESTS if (ROOT / path).exists()]
    if not manifests:
        return 0
    if (ROOT / "requirements.txt").exists():
        cmd = ["pip-audit", "-r", "requirements.txt", "--progress-spinner", "off"]
    else:
        cmd = ["pip-audit", "--progress-spinner", "off"]
    result = _run(cmd)
    if result.returncode == 0:
        return 0
    output = (result.stdout + result.stderr).strip()
    return fail_or_warn("Python dependency audit failed", [output], args.warn_only)


CHECKS = {
    "file-size": check_file_size,
    "spec-freshness": check_spec_freshness,
    "spec-changelog": check_spec_changelog,
    "adr-readme": check_adr_readme,
    "workflow-inventory": check_workflow_inventory,
    "error-handling": check_error_handling,
    "root-clutter": check_root_clutter,
    "dependency-direction": check_dependency_direction,
    "dependency-audit": check_dependency_audit,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checks", nargs="+", choices=sorted(CHECKS) + ["fast", "push"])
    parser.add_argument("--warn-only", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument(
        "--max-source-lines",
        type=int,
        default=DEFAULT_MAX_SOURCE_LINES,
    )
    args = parser.parse_args(argv)

    selected: list[str] = []
    for check in args.checks:
        if check == "fast":
            selected.extend(
                [
                    "file-size",
                    "spec-freshness",
                    "spec-changelog",
                    "adr-readme",
                    "workflow-inventory",
                    "error-handling",
                    "root-clutter",
                    "dependency-direction",
                ]
            )
        elif check == "push":
            selected.extend(["fast", "dependency-audit"])
        else:
            selected.append(check)
    if "fast" in selected:
        selected.remove("fast")
        selected.extend(
            [
                "file-size",
                "spec-freshness",
                "spec-changelog",
                "adr-readme",
                "workflow-inventory",
                "error-handling",
                "root-clutter",
                "dependency-direction",
            ]
        )

    status = 0
    for check in dict.fromkeys(selected):
        status |= CHECKS[check](args)
    if status == 0:
        logger.info("Fleet hook checks passed.")
    return status


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
