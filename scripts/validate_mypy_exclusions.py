#!/usr/bin/env python3
"""Validate mypy.ini exclusion cleanup for issue #2351.

Design-by-Contract:
  - PRE: mypy.ini exists and is parseable
  - POST: Returns 0 if config is valid, 1 otherwise
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_mypy_excludes(ini_path: Path) -> set[str]:
    """Extract excluded directory patterns from mypy.ini."""
    content = ini_path.read_text(encoding="utf-8")
    match = re.search(r"^exclude\s*=\s*\((.*?)\)", content, re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError("No exclude line found in mypy.ini")

    raw = match.group(1)
    # Split on | and strip whitespace
    patterns = {p.strip().strip("\\") for p in raw.split("|") if p.strip()}
    return patterns


def count_affected_files(patterns: set[str]) -> dict[str, int]:
    """Count Python files that match each exclusion pattern."""
    results = {}
    for pat in patterns:
        # Convert mypy regex to glob approximation for counting
        glob_pat = pat.replace(".*", "*").replace("\\.", ".").strip("^$")
        if glob_pat.endswith("/"):
            cmd = ["git", "-C", str(REPO_ROOT), "ls-files", f"{glob_pat}**/*.py"]
        else:
            cmd = ["git", "-C", str(REPO_ROOT), "ls-files", glob_pat]

        try:
            out = subprocess.run(cmd, capture_output=True, text=True, check=True)
            count = len([ln for ln in out.stdout.splitlines() if ln.strip()])
        except subprocess.CalledProcessError:
            count = 0

        results[pat] = count

    return results


def main() -> int:
    ini_path = REPO_ROOT / "mypy.ini"
    try:
        patterns = parse_mypy_excludes(ini_path)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    print("=== Mypy Exclusion Audit ===")
    print(f"Total exclusion patterns: {len(patterns)}")
    print()

    results = count_affected_files(patterns)
    total_excluded = 0
    for pat, count in sorted(results.items(), key=lambda x: -x[1]):
        total_excluded += count
        marker = "  "
        if count == 0:
            marker = "👻"  # Phantom exclusion
        elif "tools/" in pat:
            marker = "🔧"  # Tools exclusion
        elif "shared" in pat:
            marker = "📦"  # Shared package exclusion
        elif "tests" in pat:
            marker = "🧪"  # Tests exclusion
        print(f"  {marker} {pat:<60} {count:>4} files")

    print()
    print(f"Total files excluded from type-checking: {total_excluded}")

    # Validate no phantom exclusions
    phantoms = [p for p, c in results.items() if c == 0 and not p.endswith("tests/")]
    if phantoms:
        print("\n⚠️  Phantom exclusions found (0 files matched):")
        for p in phantoms:
            print(f"   - {p}")

    # Check for overlapping ruff exclusions
    ruff_path = REPO_ROOT / "ruff.toml"
    if ruff_path.exists():
        ruff_content = ruff_path.read_text()
        ruff_excluded = set()
        for line in ruff_content.splitlines():
            line = line.strip().strip('"').strip("'").strip(",")
            if line.startswith("src/") or line.startswith("tools/"):
                ruff_excluded.add(line)

        mypy_dirs = {
            p for p in patterns if p.startswith("src/") or p.startswith("tools/")
        }
        overlap = mypy_dirs & ruff_excluded
        if overlap:
            print("\n🔍 Directories excluded by BOTH mypy and ruff:")
            for o in sorted(overlap):
                print(f"   - {o}")

    print("\n✅ Audit complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
