"""Autonomous issue-solving wave runner (hardened).

Security hardening (issue #3143):
- All subprocess calls use argv lists with ``shell=False``; no command
  strings are ever built from issue titles/bodies, so quotes, semicolons,
  ampersands, newlines, and command-substitution text are passed as inert
  data and never interpreted by a shell.
- The ``--dangerously-skip-permissions`` Claude flag is opt-in only.
- Destructive/mutating git and GitHub actions (``git reset --hard``, issue
  close, ``gh pr merge --auto``) are gated behind an explicit
  ``--allow-mutations`` flag. The default mode is a dry run that prints the
  intended argv lists without mutating any state.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "WaveConfig",
    "build_claude_argv",
    "build_pr_create_argv",
    "main",
    "run_cmd",
]

REPO_ROOT = Path(__file__).resolve().parent

logger = logging.getLogger("wave_solver")


@dataclass(frozen=True)
class WaveConfig:
    """Runtime policy for a wave run.

    Attributes:
        allow_mutations: when False (default) no state-changing command is
            executed; intended actions are logged instead.
        skip_permissions: when True, pass ``--dangerously-skip-permissions``
            to the Claude CLI. Off by default.
    """

    allow_mutations: bool = False
    skip_permissions: bool = False


def run_cmd(
    argv: Sequence[str],
    cwd: Path | None = None,
    ignore_err: bool = False,
    *,
    mutating: bool = False,
    config: WaveConfig | None = None,
) -> str | None:
    """Run a command as an argv list with ``shell=False``.

    Precondition: ``argv`` is a non-empty sequence of string arguments.

    When ``mutating`` is True and ``config.allow_mutations`` is False, the
    command is logged but NOT executed (dry-run safety gate).
    """
    if not argv:
        raise ValueError("argv must be a non-empty sequence of arguments")
    if any(not isinstance(arg, str) for arg in argv):
        raise TypeError("all argv entries must be strings")

    if mutating and (config is None or not config.allow_mutations):
        logger.info("[dry-run] would run: %s", list(argv))
        return None

    try:
        result = subprocess.run(  # noqa: S603 - argv list, shell=False
            list(argv),
            shell=False,
            check=not ignore_err,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if not ignore_err:
            raise e
        return None


def build_claude_argv(prompt: str, *, skip_permissions: bool = False) -> list[str]:
    """Build the Claude CLI argv list.

    The ``prompt`` is passed as a single argv element, so any shell
    metacharacters it contains are inert. ``--dangerously-skip-permissions``
    is included only when explicitly opted in.
    """
    argv = ["claude", "-p"]
    if skip_permissions:
        argv.append("--dangerously-skip-permissions")
    argv.append(prompt)
    return argv


def build_pr_create_argv(num: int, title: str) -> list[str]:
    """Build the ``gh pr create`` argv list with the title as inert data."""
    return [
        "gh",
        "pr",
        "create",
        "--title",
        f"fix: Resolve #{num} - {title}",
        "--body",
        f"Resolves #{num}. Built via Autonomous Iterative Wave.",
        "--base",
        "main",
    ]


def _fetch_issues(repo_path: Path) -> list[dict]:
    raw_issues = run_cmd(
        ["gh", "issue", "list", "--state", "open", "--json", "number,title,body"],
        cwd=repo_path,
    )
    if not raw_issues:
        return []
    try:
        parsed = json.loads(raw_issues)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _select_issues(
    issues: list[dict], repo_path: Path, config: WaveConfig
) -> list[dict]:
    issues_to_fix: list[dict] = []
    seen_titles: set[str] = set()
    for i in issues:
        title = i.get("title", "")
        if title in seen_titles:
            # Closing an issue is a mutation: gated behind allow_mutations.
            run_cmd(
                ["gh", "issue", "close", str(i["number"])],
                cwd=repo_path,
                ignore_err=True,
                mutating=True,
                config=config,
            )
            continue
        if "[A-N Assessment]" in title:
            seen_titles.add(title)
            issues_to_fix.append(i)
    return issues_to_fix


def _build_prompt(num: int, title: str, body: str) -> str:
    return (
        f"Fix GitHub Issue #{num}: {title}. You are solving this autonomously "
        f"for high-quality, long-term codebase health. "
        f"Context: {body}. "
        f"Instructions: Open the referenced files and strictly solve the "
        f"refactoring criteria (DRY, LOD, TDD, Changeability, DbC). "
        f"Apply robust changes locally, then test with pytest before "
        f"finishing. DO NOT commit or push anything! Only yield the local "
        f"file changes."
    )


def _solve_issue(issue: dict, repo_path: Path, config: WaveConfig) -> None:
    num = issue["number"]
    title = issue["title"]
    body = issue.get("body", "").replace("\n", " ")
    branch = f"fix/a-n-issue-{num}"

    # Cleanup and branch. git reset --hard is destructive => gated.
    run_cmd(
        ["git", "reset", "--hard"],
        cwd=repo_path,
        ignore_err=True,
        mutating=True,
        config=config,
    )
    run_cmd(
        ["git", "checkout", "main"],
        cwd=repo_path,
        ignore_err=True,
        mutating=True,
        config=config,
    )
    run_cmd(
        ["git", "pull", "--rebase"],
        cwd=repo_path,
        ignore_err=True,
        mutating=True,
        config=config,
    )
    run_cmd(
        ["git", "checkout", "-b", branch],
        cwd=repo_path,
        ignore_err=True,
        mutating=True,
        config=config,
    )

    prompt = _build_prompt(num, title, body)
    claude_argv = build_claude_argv(prompt, skip_permissions=config.skip_permissions)
    # Running Claude is itself a mutation of the working tree.
    if not config.allow_mutations:
        logger.info("[dry-run] would run claude: %s", claude_argv)
        return
    try:
        subprocess.run(  # noqa: S603 - argv list, shell=False
            claude_argv, cwd=repo_path, shell=False, text=True
        )
    except OSError:
        return

    status = run_cmd(["git", "status", "--porcelain"], cwd=repo_path)
    if not status:
        return

    run_cmd(["git", "add", "-A"], cwd=repo_path, mutating=True, config=config)
    run_cmd(
        [
            "git",
            "commit",
            "-m",
            f"fix: resolve A-N assessment finding #{num} - {title}",
        ],
        cwd=repo_path,
        mutating=True,
        config=config,
    )
    run_cmd(
        ["git", "push", "-u", "origin", branch],
        cwd=repo_path,
        ignore_err=True,
        mutating=True,
        config=config,
    )

    run_cmd(
        build_pr_create_argv(num, title),
        cwd=repo_path,
        ignore_err=True,
        mutating=True,
        config=config,
    )
    # Auto-merge is destructive/irreversible => gated behind allow_mutations.
    run_cmd(
        ["gh", "pr", "merge", "--squash", "--auto"],
        cwd=repo_path,
        ignore_err=True,
        mutating=True,
        config=config,
    )
    time.sleep(2)
    run_cmd(
        ["git", "checkout", "main"],
        cwd=repo_path,
        ignore_err=True,
        mutating=True,
        config=config,
    )


def main(config: WaveConfig | None = None) -> None:
    """Run a single wave. Defaults to dry-run (no mutations)."""
    if config is None:
        config = _parse_args()
    repo_path = REPO_ROOT
    issues = _fetch_issues(repo_path)
    if not issues:
        return
    issues_to_fix = _select_issues(issues, repo_path, config)
    for issue in issues_to_fix:
        _solve_issue(issue, repo_path, config)


def _parse_args(argv: Sequence[str] | None = None) -> WaveConfig:
    parser = argparse.ArgumentParser(description="Autonomous issue-solving wave.")
    parser.add_argument(
        "--allow-mutations",
        action="store_true",
        help="Execute state-changing git/gh actions. Default is dry-run.",
    )
    parser.add_argument(
        "--skip-permissions",
        action="store_true",
        help="Pass --dangerously-skip-permissions to the Claude CLI.",
    )
    args = parser.parse_args(argv)
    return WaveConfig(
        allow_mutations=args.allow_mutations,
        skip_permissions=args.skip_permissions,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
