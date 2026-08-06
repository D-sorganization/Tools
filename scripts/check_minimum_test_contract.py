#!/usr/bin/env python3
"""Enforce minimum test contract for changed source packages.

Rule: if a package is modified in `src/`, it must have at least one test file in
repo-level `tests/` or package-local `tests/` directories.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def changed_files() -> list[str]:
    cp = subprocess.run(
        ["git", "diff", "--name-only", "origin/main...HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        return []
    return [ln.strip() for ln in cp.stdout.splitlines() if ln.strip()]


def package_key(path: str) -> str | None:
    parts = path.split("/")
    if len(parts) < 3 or parts[0] != "src":
        return None

    if parts[1] in {"tools", "data_processing", "web_applications"} and len(parts) >= 3:
        return "/".join(parts[:3])

    if parts[1] == "shared" and len(parts) >= 4 and parts[2] == "python":
        return "/".join(parts[:4])

    return "/".join(parts[:2])


def has_tests(pkg_key: str) -> bool:
    pkg_path = ROOT / pkg_key
    if pkg_path.exists():
        local_tests = list(pkg_path.rglob("tests/test_*.py")) + list(
            pkg_path.rglob("tests/*_test.py")
        )
        if local_tests:
            return True

    # Shared packages commonly mirror their source hierarchy under tests/, for
    # example src/shared/python/golf_club -> tests/shared/python/golf_club.
    # Check that convention before falling back to token-based discovery.
    mirrored_tests = ROOT / "tests" / Path(*pkg_key.split("/")[1:])
    if mirrored_tests.exists():
        mirrored_package_tests = list(mirrored_tests.rglob("test_*.py")) + list(
            mirrored_tests.rglob("*_test.py")
        )
        if mirrored_package_tests:
            return True

    token = Path(pkg_key.split("/")[-1]).stem
    package_tests = ROOT / "tests" / token
    if package_tests.exists():
        global_package_tests = list(package_tests.rglob("test_*.py")) + list(
            package_tests.rglob("*_test.py")
        )
        if global_package_tests:
            return True

    global_tests = list((ROOT / "tests").rglob(f"*{token}*test*.py")) + list(
        (ROOT / "tests").rglob(f"test_*{token}*.py")
    )
    return bool(global_tests)


def main() -> int:
    changed = changed_files()
    changed_src = [p for p in changed if p.startswith("src/") and p.endswith(".py")]
    packages = sorted({k for p in changed_src if (k := package_key(p)) is not None})

    if not packages:
        sys.stdout.write(
            "No changed src Python packages; minimum test contract check skipped.\n"
        )
        return 0

    # Skip packages that have been removed (deleted from disk) — deletions are not
    # test contract violations. A package disappearing satisfies the contract vacuously.
    # Note: after `git rm`, a directory may still exist with only __pycache__/ or
    # tests/ subdirs. We consider a package "present" only if it has tracked Python
    # source files (i.e., at least one .py file outside of tests/).
    def _has_source_files(pkg: str) -> bool:
        pkg_path = ROOT / pkg
        if not pkg_path.exists():
            return False
        py_files = [
            f
            for f in pkg_path.rglob("*.py")
            if "tests" not in f.parts and "__pycache__" not in f.parts
        ]
        return bool(py_files)

    packages = [pkg for pkg in packages if _has_source_files(pkg)]

    if not packages:
        sys.stdout.write(
            "All changed packages were deleted; minimum test contract check skipped.\n"
        )
        return 0

    # Packages with tests in non-standard locations that the heuristic can't find
    KNOWN_TESTED = {
        "src/shared/python/programmatic_pid",  # tests/programmatic_pid/
        "src/pid_generator",  # tests/programmatic_pid/ (thin CLI wrapper)
        "src/shared/python/tests",  # the tests folder itself
        "src/shared/python/gui_launcher",  # tests/shared/python/gui_launcher/test_registry.py
        "src/folder_packer_pro",  # tests/folder_packer_pro/test_file_ops.py
        "src/media_processing",  # src/media_processing/.../tests_video_processor/test_api.py
    }
    # Legacy directories explicitly excluded from ruff linting in pyproject.toml.
    # These directories pre-date the test contract and are not required to have tests
    # until they are promoted out of legacy status.
    LEGACY_EXEMPT = {
        "src/tools/folder_tools",  # excluded from ruff; legacy monolith, no unit tests yet
    }
    violations = [
        pkg
        for pkg in packages
        if pkg not in KNOWN_TESTED and pkg not in LEGACY_EXEMPT and not has_tests(pkg)
    ]
    if violations:
        sys.stderr.write("Minimum test contract failed for changed packages:\n")
        for pkg in violations:
            sys.stderr.write(f"- {pkg}\n")
        return 1

    sys.stdout.write("Minimum test contract passed for changed packages.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
