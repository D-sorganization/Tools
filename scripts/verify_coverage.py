#!/usr/bin/env python
"""Coverage verification script for the Sidekick package (issues #3032, #3033).

Runs pytest with coverage instrumentation on the sidekick package and
enforces:
  - Overall sidekick package coverage >= 50%
  - Per-module coverage >= 70% for the 7 public facade modules

Exit codes:
  0 — All coverage thresholds met.
  1 — One or more thresholds failed.

Usage::

    python scripts/verify_coverage.py [--html]

Options:
    --html      Also generate an HTML coverage report in htmlcov/.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
SIDEKICK_SRC = REPO_ROOT / "src" / "shared" / "python" / "sidekick"

# Facade modules that must individually reach >= 70%
REQUIRED_MODULES: list[str] = [
    "bootstrap",
    "latex_renderer",
    "notes_store",
    "notes_tab",
    "selected_tab_panel",
    "symbolic_engine",
    "tab_context_menu",
]

PACKAGE_THRESHOLD = 50  # %
MODULE_THRESHOLD = 70  # %


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report.",
    )
    return parser.parse_args()


def run_pytest_with_coverage(html: bool) -> tuple[int, str]:
    """Run pytest with coverage JSON output and return (returncode, json_path)."""
    json_path = REPO_ROOT / ".coverage_report.json"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--tb=short",
        "-q",
        f"--cov={SIDEKICK_SRC}",
        "--cov-report=json:" + str(json_path),
        "--cov-report=term-missing:skip-covered",
        "tests/unit/sidekick/",
        "tests/test_sidekick_public_api_stability.py",
        "-m",
        "not gui",
    ]
    if html:
        cmd.append("--cov-report=html")
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))  # noqa: S603
    return result.returncode, str(json_path)


def load_coverage_json(json_path: str) -> dict:
    """Load the JSON coverage report."""
    path = Path(json_path)
    if not path.is_file():
        log.error("Coverage JSON not found at %s", json_path)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def check_thresholds(coverage_data: dict) -> bool:
    """Check overall and per-module thresholds. Return True if all pass."""
    totals = coverage_data.get("totals", {})
    overall_pct: float = totals.get("percent_covered", 0.0)

    failures: list[str] = []

    if overall_pct < PACKAGE_THRESHOLD:
        failures.append(
            f"Overall sidekick coverage {overall_pct:.1f}% < {PACKAGE_THRESHOLD}% required."
        )

    files: dict = coverage_data.get("files", {})

    for module_name in REQUIRED_MODULES:
        matched: dict | None = None
        for key, data in files.items():
            if key.replace("\\", "/").endswith(f"sidekick/{module_name}.py"):
                matched = data
                break
        if matched is None:
            failures.append(
                f"Module {module_name!r} not found in coverage report. "
                "Ensure it is imported by at least one test."
            )
            continue
        summary = matched.get("summary", {})
        pct: float = summary.get("percent_covered", 0.0)
        if pct < MODULE_THRESHOLD:
            failures.append(
                f"  sidekick/{module_name}.py: {pct:.1f}% < {MODULE_THRESHOLD}% required."
            )

    if failures:
        print("\n❌  Coverage threshold violations:")
        for msg in failures:
            print(f"   {msg}")
        print()
        return False

    print(
        f"\n✅  All coverage thresholds met "
        f"(overall: {overall_pct:.1f}% >= {PACKAGE_THRESHOLD}%)."
    )
    return True


def main() -> int:
    """Entry point. Returns exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    pytest_rc, json_path = run_pytest_with_coverage(html=args.html)
    if pytest_rc != 0:
        log.error("pytest exited with code %d", pytest_rc)

    coverage_data = load_coverage_json(json_path)
    passed = check_thresholds(coverage_data)

    return 0 if passed and pytest_rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
