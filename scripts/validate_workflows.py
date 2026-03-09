#!/usr/bin/env python3
"""Validate GitHub Actions workflow files for basic structural correctness."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def iter_workflows(root: Path) -> list[Path]:
    return sorted(root.glob("*.yml")) + sorted(root.glob("*.yaml"))


def validate_workflow(path: Path) -> list[str]:
    errors: list[str] = []

    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except Exception as exc:  # pragma: no cover - surfaced directly in CI
        return [f"{path}: YAML parse error: {exc}"]

    if not isinstance(data, dict):
        return [f"{path}: expected a top-level mapping"]

    jobs = data.get("jobs")
    if jobs is None:
        errors.append(f"{path}: missing top-level 'jobs'")
    elif not isinstance(jobs, dict) or not jobs:
        errors.append(f"{path}: expected a non-empty 'jobs' mapping")

    return errors


def main() -> int:
    workflows_dir = Path(".github/workflows")
    workflow_paths = iter_workflows(workflows_dir)

    if not workflow_paths:
        print("No workflow files found.", file=sys.stderr)
        return 1

    errors: list[str] = []
    for workflow_path in workflow_paths:
        errors.extend(validate_workflow(workflow_path))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"Validated {len(workflow_paths)} workflow files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
