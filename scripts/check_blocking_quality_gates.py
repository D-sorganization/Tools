#!/usr/bin/env python3
"""Verify core PR quality gates remain blocking."""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in lean CI images
    yaml = None

REQUIRED_BLOCKING_STEPS = {
    "Lint (Ruff)": "ruff check",
    "Format Check (ruff-format)": "ruff format --check",
    "Type Check (Mypy)": "mypy",
}


def _step_run(step: dict[str, object]) -> str:
    value = step.get("run", "")
    return value if isinstance(value, str) else ""


def _command_masks_failure(run: str, command: str) -> bool:
    if command == "mypy":
        command_re = re.compile(r"(^|[\s|])mypy(\s|$)")
        return any(
            "|| true" in line and command_re.search(line) for line in run.splitlines()
        )
    return any(command in line and "|| true" in line for line in run.splitlines())


def _validate_ci_standard_text(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    for step_name, command in REQUIRED_BLOCKING_STEPS.items():
        pattern = re.compile(
            rf"(?ms)^\s*-\s+name:\s+{re.escape(step_name)}\s*$"
            rf"(?P<body>.*?)(?=^\s*-\s+name:|\n\s{{2}}\S|\Z)"
        )
        match = pattern.search(text)
        if match is None:
            errors.append(f"{path}: missing blocking step {step_name!r}")
            continue
        body = match.group("body")
        if re.search(r"^\s*continue-on-error:\s*true\s*$", body, re.MULTILINE):
            errors.append(f"{path}: {step_name!r} must not use continue-on-error")
        if command not in body:
            errors.append(f"{path}: {step_name!r} must run {command!r}")
        if _command_masks_failure(body, command):
            errors.append(f"{path}: {step_name!r} must not mask failures with || true")
    return errors


def validate_ci_standard(path: Path) -> list[str]:
    if yaml is None:
        return _validate_ci_standard_text(path)

    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    jobs = workflow.get("jobs", {})
    quality_gate = jobs.get("quality-gate", {}) if isinstance(jobs, dict) else {}
    steps = quality_gate.get("steps", []) if isinstance(quality_gate, dict) else []
    if not isinstance(steps, list):
        return [f"{path}: quality-gate steps must be a list"]

    errors: list[str] = []
    for step_name, command in REQUIRED_BLOCKING_STEPS.items():
        matches = [
            step
            for step in steps
            if isinstance(step, dict) and step.get("name") == step_name
        ]
        if not matches:
            errors.append(f"{path}: missing blocking step {step_name!r}")
            continue
        step = matches[0]
        if step.get("continue-on-error") is True:
            errors.append(f"{path}: {step_name!r} must not use continue-on-error")
        run = _step_run(step)
        if command not in run:
            errors.append(f"{path}: {step_name!r} must run {command!r}")
        if _command_masks_failure(run, command):
            errors.append(f"{path}: {step_name!r} must not mask failures with || true")
    return errors


def main() -> int:
    errors = validate_ci_standard(Path(".github/workflows/ci-standard.yml"))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("Blocking quality gate policy passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
