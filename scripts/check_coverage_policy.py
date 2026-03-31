#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _pct(x: float) -> float:
    return round(x * 100.0, 2)


def _is_omitted(filename: str, omit_patterns: list[str]) -> bool:
    """Return True if the file path matches any omit glob pattern."""
    normalized = filename.replace("\\", "/")
    for pattern in omit_patterns:
        # Support both path-prefix patterns and fnmatch glob patterns
        norm_pattern = pattern.replace("\\", "/")
        if fnmatch.fnmatch(normalized, norm_pattern):
            return True
        # Also match against just the basename
        basename = normalized.rsplit("/", 1)[-1]
        if fnmatch.fnmatch(basename, norm_pattern):
            return True
    return False


def parse_coverage(
    coverage_file: Path,
    tracked_prefixes: list[str],
    omit_patterns: list[str] | None = None,
    include_prefixes: list[str] | None = None,
) -> dict[str, object]:
    root = ET.parse(coverage_file).getroot()

    omit_patterns = omit_patterns or []
    per_prefix = {p: {"covered": 0, "valid": 0} for p in tracked_prefixes}
    total_covered = 0
    total_valid = 0

    for cls in root.findall(".//class"):
        filename = cls.attrib.get("filename", "")
        normalized = filename.replace("\\", "/")

        # When include_prefixes is specified, only count files under those prefixes
        if include_prefixes and not any(
            normalized.startswith(p) for p in include_prefixes
        ):
            # Still track per-package metrics for tracked_prefixes even if excluded
            # from total, so package-level thresholds are not affected by the filter.
            lines = cls.findall("./lines/line")
            valid = len(lines)
            covered = sum(1 for ln in lines if int(ln.attrib.get("hits", "0")) > 0)
            for prefix in tracked_prefixes:
                if normalized.startswith(prefix):
                    per_prefix[prefix]["covered"] += covered
                    per_prefix[prefix]["valid"] += valid
            continue

        if _is_omitted(filename, omit_patterns):
            continue

        lines = cls.findall("./lines/line")
        valid = len(lines)
        covered = sum(1 for ln in lines if int(ln.attrib.get("hits", "0")) > 0)
        total_covered += covered
        total_valid += valid
        for prefix in tracked_prefixes:
            if normalized.startswith(prefix):
                per_prefix[prefix]["covered"] += covered
                per_prefix[prefix]["valid"] += valid

    total_line_rate = (total_covered / total_valid) if total_valid else 0.0

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
    omit = list(policy.get("omit_patterns", []))
    include = list(policy.get("include_prefixes", []))
    current = parse_coverage(Path(args.coverage_file), tracked, omit, include or None)

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
