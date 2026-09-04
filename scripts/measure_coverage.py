#!/usr/bin/env python3
"""
Coverage measurement and baseline comparison script.

Measures test coverage and compares against baseline to detect regressions.
Generates human-readable reports and JSON output for CI integration.

Usage:
    python scripts/measure_coverage.py [--baseline-file config/coverage_baseline.json] [--output-dir coverage_reports]
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_coverage_xml(xml_file: Path, tracked_prefixes: list[str]) -> dict[str, Any]:
    """Parse coverage.xml and extract per-package coverage metrics."""
    root = ET.parse(xml_file).getroot()
    total_line_rate = float(root.attrib.get("line-rate", "0"))

    per_prefix = {p: {"covered": 0, "valid": 0} for p in tracked_prefixes}

    for cls in root.findall(".//class"):
        filename = cls.attrib.get("filename", "")
        # Reconstruct src/ prefix
        full_path = f"src/{filename}"

        lines = cls.findall("./lines/line")
        valid = len(lines)
        covered = sum(1 for ln in lines if int(ln.attrib.get("hits", "0")) > 0)

        for prefix in tracked_prefixes:
            if full_path.startswith(prefix):
                per_prefix[prefix]["covered"] += covered
                per_prefix[prefix]["valid"] += valid

    package_pct: dict[str, float] = {}
    for prefix, stats in per_prefix.items():
        valid = stats["valid"]
        pct = round(((stats["covered"] / valid) * 100) if valid else 0.0, 2)
        package_pct[prefix] = pct

    return {
        "total_percent": round(total_line_rate * 100, 2),
        "package_percent": package_pct,
    }


def compare_coverage(
    current: dict[str, Any], baseline: dict[str, Any], policy: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """
    Compare current coverage against baseline and policy thresholds.

    Returns:
        (report_dict, failures_list)
    """
    failures: list[str] = []
    report: dict[str, Any] = {
        "current": current,
        "baseline": baseline,
        "timestamp": datetime.now().isoformat(),
        "policy_check": {},
    }

    # The floor lives in pyproject only (Tools #4913); never in the policy file.
    from check_coverage_policy import coverage_floor

    min_total = coverage_floor()
    max_drop = float(policy.get("max_total_drop_percent", 0.0))
    baseline_total = float(baseline.get("total_percent", 0.0))

    current_total = float(current["total_percent"])

    # Check total coverage thresholds
    if current_total < min_total:
        failures.append(f"Total coverage {current_total}% below minimum {min_total}%")
    if baseline_total > 0 and current_total < (baseline_total - max_drop):
        failures.append(
            f"Total coverage {current_total}% regressed beyond allowed drop "
            f"({baseline_total}% - {max_drop}% = {baseline_total - max_drop}%)"
        )

    report["policy_check"]["total"] = {
        "current": current_total,
        "baseline": baseline_total,
        "minimum": min_total,
        "max_allowed_drop": max_drop,
        "passed": not any("Total coverage" in f for f in failures),
    }

    # Check package thresholds
    pkg_current: dict[str, float] = current.get("package_percent", {})
    pkg_min: dict[str, float] = policy.get("tracked_packages", {})

    pkg_results = {}
    for pkg, threshold in pkg_min.items():
        cur = float(pkg_current.get(pkg, 0.0))
        passed = cur >= float(threshold)
        pkg_results[pkg] = {
            "current": cur,
            "threshold": threshold,
            "passed": passed,
        }
        if not passed:
            failures.append(
                f"Package {pkg} coverage {cur}% below threshold {threshold}%"
            )

    report["policy_check"]["packages"] = pkg_results

    return report, failures


def main() -> int:
    """Main entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--coverage-file",
        type=Path,
        default=Path("coverage.xml"),
        help="Path to coverage.xml file",
    )
    ap.add_argument(
        "--baseline-file",
        type=Path,
        default=Path("config/coverage_baseline.json"),
        help="Path to baseline JSON file",
    )
    ap.add_argument(
        "--policy-file",
        type=Path,
        default=Path("config/coverage_policy.json"),
        help="Path to coverage policy JSON file",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("coverage_reports"),
        help="Output directory for coverage reports",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if coverage regresses at all (even within allowed drop)",
    )
    args = ap.parse_args()

    # Check files exist
    if not args.coverage_file.exists():
        print(f"ERROR: Coverage file not found: {args.coverage_file}", file=sys.stderr)
        return 1
    if not args.baseline_file.exists():
        print(f"ERROR: Baseline file not found: {args.baseline_file}", file=sys.stderr)
        return 1
    if not args.policy_file.exists():
        print(f"ERROR: Policy file not found: {args.policy_file}", file=sys.stderr)
        return 1

    # Load baseline and policy
    baseline = json.loads(args.baseline_file.read_text(encoding="utf-8"))
    policy = json.loads(args.policy_file.read_text(encoding="utf-8"))

    # Parse coverage
    tracked = list(policy.get("tracked_packages", {}).keys())
    current = parse_coverage_xml(args.coverage_file, tracked)

    # Compare
    report, failures = compare_coverage(current, baseline, policy)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write report JSON
    report_file = args.output_dir / "coverage_report.json"
    report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Print summary
    print("=" * 70)
    print("Coverage Measurement Report")
    print("=" * 70)
    print(f"Total Coverage: {current['total_percent']}%")
    print(f"Baseline:       {baseline.get('total_percent', 'N/A')}%")

    if report["policy_check"]["packages"]:
        print("\nPackage Coverage:")
        for pkg, stats in report["policy_check"]["packages"].items():
            status = "PASS" if stats["passed"] else "FAIL"
            print(
                f"  [{status}] {pkg}: {stats['current']}% "
                f"(threshold: {stats['threshold']}%)"
            )

    print(f"\nReport written to: {report_file}")

    if failures:
        print("\n" + "=" * 70)
        print("FAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        print("=" * 70)
        if args.strict:
            return 1
        # Non-strict mode: warn but don't fail
        return 0

    print("\nAll coverage checks passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
