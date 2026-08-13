#!/usr/bin/env python3
"""Fail when GitHub Actions workflows can route to hosted runners."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

WORKFLOW_DIR = Path(".github") / "workflows"
BANNED = (
    "force_cloud",
    "mode=cloud",
    "Routing to GitHub-hosted",
    "using GitHub-hosted",
    "runner=ubuntu-latest",
    "runner=windows-latest",
    "runner=macos-latest",
)

LEGACY_HOSTED_RUNNER_ALLOWLIST = {
    ".github/workflows/Jules-Assessment-Generator.yml",
    ".github/workflows/Jules-Assessment-Remediator.yml",
    ".github/workflows/Jules-Code-Quality-Reviewer.yml",
    ".github/workflows/Jules-Completist.yml",
    ".github/workflows/Jules-Comprehensive-Assessment.yml",
    ".github/workflows/Jules-Critics-Comments.yml",
    ".github/workflows/Jules-Laymans-Terms-Writer.yml",
    ".github/workflows/file-size-budget.yml",
    ".github/workflows/local-only-runner-guard.yml",
    ".github/workflows/nightly-full-repo-quality.yml",
}
HOSTED_RUNNER_ALLOWLIST = {
    (".github/workflows/ci-standard.yml", "quality-gate"),
    (".github/workflows/rate-web-playwright.yml", "fork-production-worker-e2e"),
}
HOSTED_RUNNER = re.compile(r"^(ubuntu|macos|windows)(-latest|-\d+(?:\.\d+)*)$")


def _hosted_runner_failures(path: Path, text: str) -> list[str]:
    data = yaml.safe_load(text)
    jobs = data.get("jobs", {}) if isinstance(data, dict) else {}
    failures: list[str] = []
    for job_id, job in jobs.items() if isinstance(jobs, dict) else ():
        if not isinstance(job, dict):
            continue
        runs_on = job.get("runs-on")
        labels = runs_on if isinstance(runs_on, list) else [runs_on]
        for label in labels:
            if not isinstance(label, str) or not HOSTED_RUNNER.fullmatch(label):
                continue
            if (path.as_posix(), str(job_id)) in HOSTED_RUNNER_ALLOWLIST:
                continue
            failures.append(f"{path}: job {job_id!r} uses hosted runner {label!r}")
    return failures


def main() -> int:
    failures: list[str] = []
    if not WORKFLOW_DIR.exists():
        return 0

    for path in sorted(WORKFLOW_DIR.rglob("*")):
        if path.suffix not in {".yml", ".yaml"}:
            continue
        if path.as_posix() in LEGACY_HOSTED_RUNNER_ALLOWLIST:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8-sig")
        failures.extend(_hosted_runner_failures(path, text))
        for line_number, line in enumerate(text.splitlines(), start=1):
            for token in BANNED:
                if token in line:
                    failures.append(
                        f"{path}:{line_number}: banned hosted-runner token {token!r}"
                    )

    if failures:
        print(
            "GitHub-hosted runner routing is forbidden. "
            "Use local self-hosted runners only."
        )
        print("\n".join(failures))
        return 1

    print("Workflow runner routing is local-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
