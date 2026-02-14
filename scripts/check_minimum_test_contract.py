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

    token = pkg_key.split("/")[-1]
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

    violations = [pkg for pkg in packages if not has_tests(pkg)]
    if violations:
        sys.stderr.write("Minimum test contract failed for changed packages:\n")
        for pkg in violations:
            sys.stderr.write(f"- {pkg}\n")
        return 1

    sys.stdout.write("Minimum test contract passed for changed packages.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
