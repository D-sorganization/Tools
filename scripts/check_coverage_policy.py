#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _pct(x: float) -> float:
    return round(x * 100.0, 2)


def parse_coverage(
    coverage_file: Path, tracked_prefixes: list[str]
) -> dict[str, object]:
    root = ET.parse(coverage_file).getroot()
    total_line_rate = float(root.attrib.get("line-rate", "0"))

    per_prefix = {p: {"covered": 0, "valid": 0} for p in tracked_prefixes}

    for cls in root.findall(".//class"):
        filename = cls.attrib.get("filename", "")
        lines = cls.findall("./lines/line")
        valid = len(lines)
        covered = sum(1 for ln in lines if int(ln.attrib.get("hits", "0")) > 0)
        for prefix in tracked_prefixes:
            if filename.startswith(prefix):
                per_prefix[prefix]["covered"] += covered
                per_prefix[prefix]["valid"] += valid

    package_pct: dict[str, float] = {}
    for prefix, stats in per_prefix.items():
        valid = stats["valid"]
        package_pct[prefix] = _pct((stats["covered"] / valid) if valid else 0.0)

    return {"total_percent": _pct(total_line_rate), "package_percent": package_pct}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage-file", default="coverage.xml")
    ap.add_argument("--policy-file", default="config/coverage_policy.json")
    ap.add_argument("--baseline-file", default="config/coverage_baseline.json")
    ap.add_argument("--output-json", default="coverage_trend.json")
    args = ap.parse_args()

    policy = json.loads(Path(args.policy_file).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline_file).read_text(encoding="utf-8"))

    tracked = list(policy.get("tracked_packages", {}).keys())
    current = parse_coverage(Path(args.coverage_file), tracked)

    min_total = float(policy.get("minimum_total_percent", 0.0))
    max_drop = float(policy.get("max_total_drop_percent", 0.0))
    baseline_total = float(baseline.get("total_percent", 0.0))

    failures: list[str] = []
    total = float(current["total_percent"])
    if total < min_total:
        failures.append(f"total coverage {total}% below minimum {min_total}%")
    if total < (baseline_total - max_drop):
        failures.append(
            f"total coverage {total}% regressed beyond allowed drop ({baseline_total}% -> {baseline_total - max_drop}%)"
        )

    pkg_current: dict[str, float] = current["package_percent"]  # type: ignore[assignment]
    pkg_min: dict[str, float] = policy.get("tracked_packages", {})
    for pkg, threshold in pkg_min.items():
        cur = float(pkg_current.get(pkg, 0.0))
        if cur < float(threshold):
            failures.append(
                f"package {pkg} coverage {cur}% below threshold {threshold}%"
            )

    Path(args.output_json).write_text(
        json.dumps(current, indent=2) + "\n", encoding="utf-8"
    )

    sys.stdout.write("Coverage policy evaluation:\n")
    sys.stdout.write(f"- total: {total}%\n")
    for pkg, value in pkg_current.items():
        sys.stdout.write(f"- {pkg}: {value}%\n")

    if failures:
        sys.stderr.write("Coverage policy failed:\n")
        for item in failures:
            sys.stderr.write(f"- {item}\n")
        return 1

    sys.stdout.write("Coverage policy passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
