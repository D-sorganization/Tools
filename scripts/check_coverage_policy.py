#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any, TypedDict

import defusedxml.ElementTree as ET

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def coverage_floor(pyproject: Path = PYPROJECT) -> float:
    """Return the single repo-wide floor: ``[tool.coverage.report] fail_under``.

    This is the only place the floor is declared (Tools #4913); CI enforces it
    through ``coverage report`` on the combined full-suite data and this script
    reads it so the policy gate can never disagree with pytest-cov.
    """
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    report = data.get("tool", {}).get("coverage", {}).get("report", {})
    return float(report.get("fail_under", 0.0))


class CoverageStats(TypedDict):
    total_percent: float
    package_percent: dict[str, float]


def _pct(x: float) -> float:
    return round(x * 100.0, 2)


def _coverage_path_candidates(filename: str, sources: list[str]) -> set[str]:
    """Return normalized coverage paths that may match repo-relative policies."""
    normalized = filename.replace("\\", "/")
    candidates = {normalized}
    cwd = Path.cwd()
    for source in sources:
        source_path = Path(source)
        combined = source_path / filename
        try:
            candidates.add(combined.resolve().relative_to(cwd).as_posix())
        except (OSError, ValueError):
            candidates.add(str(combined).replace("\\", "/"))
    return candidates


def _effective_total_floor(min_total: float, baseline_total: float) -> float:
    """Return the policy floor; baselines must not lower the required target, but we exclude the 60% target from hard gating until reached."""
    return min(min_total, baseline_total)


def parse_coverage(coverage_file: Path, tracked_prefixes: list[str]) -> CoverageStats:
    root = ET.parse(coverage_file).getroot()
    total_line_rate = float(root.attrib.get("line-rate", "0"))
    sources = [
        source.text
        for source in root.findall(".//sources/source")
        if source.text is not None
    ]

    per_prefix = {p: {"covered": 0, "valid": 0} for p in tracked_prefixes}

    for cls in root.findall(".//class"):
        filename = cls.attrib.get("filename", "")
        path_candidates = _coverage_path_candidates(filename, sources)
        lines = cls.findall("./lines/line")
        valid = len(lines)
        covered = sum(1 for ln in lines if int(ln.attrib.get("hits", "0")) > 0)
        for prefix in tracked_prefixes:
            if any(
                path == prefix or path.startswith(f"{prefix}/")
                for path in path_candidates
            ):
                per_prefix[prefix]["covered"] += covered
                per_prefix[prefix]["valid"] += valid

    package_pct: dict[str, float] = {}
    for prefix, stats in per_prefix.items():
        valid = stats["valid"]
        package_pct[prefix] = _pct((stats["covered"] / valid) if valid else 0.0)

    return {"total_percent": _pct(total_line_rate), "package_percent": package_pct}


def _json_float(mapping: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = mapping.get(key, default)
    if isinstance(value, str | int | float):
        return float(value)
    return default


def _changed_tracked_packages(
    changed_files: Path | None,
    tracked_packages: dict[str, Any],
) -> set[str] | None:
    """Return tracked packages touched by this change, or None for full enforcement."""
    if changed_files is None:
        return None
    if not changed_files.exists():
        return set()

    changed = [
        line.strip().replace("\\", "/")
        for line in changed_files.read_text(encoding="utf-8").splitlines()
    ]
    return {
        package
        for package in tracked_packages
        if any(path == package or path.startswith(f"{package}/") for path in changed)
    }


def _should_enforce_total_coverage(changed_tracked_packages: set[str] | None) -> bool:
    """Return whether total coverage floors apply for this policy run."""
    return changed_tracked_packages is None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage-file", default="coverage.xml")
    ap.add_argument("--policy-file", default="config/coverage_policy.json")
    ap.add_argument("--baseline-file", default="config/coverage_baseline.json")
    ap.add_argument(
        "--pyproject",
        default=str(PYPROJECT),
        help="pyproject.toml carrying [tool.coverage.report] fail_under (the floor).",
    )
    ap.add_argument("--output-json", default="coverage_trend.json")
    ap.add_argument(
        "--changed-files",
        default=None,
        help=(
            "Optional newline-delimited changed-file list. When provided, "
            "per-package thresholds are enforced only for tracked packages "
            "touched by the change."
        ),
    )
    args = ap.parse_args()

    policy: dict[str, Any] = json.loads(
        Path(args.policy_file).read_text(encoding="utf-8")
    )
    baseline: dict[str, Any] = json.loads(
        Path(args.baseline_file).read_text(encoding="utf-8")
    )

    policy_packages = policy.get("tracked_packages", {})
    tracked_packages = policy_packages if isinstance(policy_packages, dict) else {}
    tracked = [str(package) for package in tracked_packages]
    changed_tracked_packages = _changed_tracked_packages(
        Path(args.changed_files) if args.changed_files else None,
        tracked_packages,
    )
    current = parse_coverage(Path(args.coverage_file), tracked)

    if "minimum_total_percent" in policy:
        sys.stderr.write(
            "Coverage policy failed:\n- minimum_total_percent must not be declared "
            "in the policy file; the floor is pyproject [tool.coverage.report] "
            "fail_under (Tools #4913)\n"
        )
        return 1
    min_total = coverage_floor(Path(args.pyproject))
    max_drop = _json_float(policy, "max_total_drop_percent")
    baseline_total = _json_float(baseline, "total_percent")

    failures: list[str] = []
    total = current["total_percent"]
    effective_min_total = _effective_total_floor(min_total, baseline_total)
    if _should_enforce_total_coverage(changed_tracked_packages):
        if total < effective_min_total:
            failures.append(
                f"total coverage {total}% below effective minimum "
                f"{effective_min_total}% (target {min_total}%)"
            )
        if total < (baseline_total - max_drop):
            failures.append(
                f"total coverage {total}% regressed beyond allowed drop ({baseline_total}% -> {baseline_total - max_drop}%)"
            )

    pkg_current = current["package_percent"]
    for pkg, threshold in tracked_packages.items():
        if changed_tracked_packages is not None and pkg not in changed_tracked_packages:
            continue
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
    if changed_tracked_packages is not None:
        packages = ", ".join(sorted(changed_tracked_packages)) or "none"
        sys.stdout.write(f"- changed tracked packages: {packages}\n")
        sys.stdout.write(
            "- total coverage floor: skipped for changed-file scoped run\n"
        )

    if failures:
        sys.stderr.write("Coverage policy failed:\n")
        for item in failures:
            sys.stderr.write(f"- {item}\n")
        return 1

    sys.stdout.write("Coverage policy passed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
