#!/usr/bin/env python3
"""
Setup script for installing pre-commit and pre-push hooks.

This script:
1. Installs pre-commit if not present
2. Installs pre-commit hooks
3. Installs pre-push hooks
4. Installs hook dependencies
5. Registers local-only git merge drivers (e.g. module-inventory-regen)
6. Verifies the installation

Usage:
    python scripts/setup_hooks.py
"""

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    logger.info(f"  Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def check_pre_commit_installed() -> bool:
    """Check if pre-commit is installed."""
    try:
        result = run_command(["pre-commit", "--version"], check=False)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def install_pre_commit() -> None:
    """Install pre-commit via pip."""
    logger.info("\n[1/5] Installing pre-commit...")
    if check_pre_commit_installed():
        logger.info("  pre-commit is already installed")
    else:
        run_command([sys.executable, "-m", "pip", "install", "pre-commit"])
        logger.info("  pre-commit installed successfully")


def install_hooks() -> None:
    """Install pre-commit hooks."""
    logger.info("\n[2/5] Installing pre-commit hooks...")
    run_command(["pre-commit", "install"])
    logger.info("  pre-commit hooks installed")


def install_push_hooks() -> None:
    """Install pre-push hooks."""
    logger.info("\n[3/5] Installing pre-push hooks...")
    run_command(["pre-commit", "install", "--hook-type", "pre-push"])
    logger.info("  pre-push hooks installed")


def install_dev_dependencies() -> None:
    """Install development dependencies for hooks."""
    logger.info("\n[4/5] Installing hook dependencies...")
    deps = [
        "ruff>=0.14.0",
        "mypy>=1.13.0",
        "bandit>=1.7.0",
        "pip-audit>=2.7.0",
        "types-requests",
        "types-PyYAML",
        "pydantic",
    ]
    run_command([sys.executable, "-m", "pip", "install"] + deps)
    logger.info("  Dependencies installed")


def install_merge_drivers() -> None:
    """Register local-only git merge drivers (e.g. module-inventory-regen).

    .gitattributes can only name a driver; the command it runs is local
    git config that has to be set up per clone/worktree. See
    scripts/git/install_merge_drivers.py for why.
    """
    logger.info("\n[5/5] Registering git merge drivers...")
    repo_root = Path(__file__).resolve().parent.parent
    run_command(
        [
            sys.executable,
            str(repo_root / "scripts" / "git" / "install_merge_drivers.py"),
        ]
    )
    logger.info("  Merge drivers registered")


def verify_installation() -> None:
    """Verify hooks are installed correctly."""
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION")
    logger.info("=" * 60)

    git_hooks_dir = Path(".git/hooks")
    pre_commit_hook = git_hooks_dir / "pre-commit"
    pre_push_hook = git_hooks_dir / "pre-push"

    if pre_commit_hook.exists():
        logger.info(f"  [OK] pre-commit hook: {pre_commit_hook}")
    else:
        logger.info(f"  [MISSING] pre-commit hook: {pre_commit_hook}")

    if pre_push_hook.exists():
        logger.info(f"  [OK] pre-push hook: {pre_push_hook}")
    else:
        logger.info(f"  [MISSING] pre-push hook: {pre_push_hook}")


def print_summary() -> None:
    """Print usage summary."""
    logger.info("\n" + "=" * 60)
    logger.info("HOOK SUMMARY")
    logger.info("=" * 60)
    logger.info("""
PRE-COMMIT (runs on every commit, <15 seconds):
  - ruff (lint + auto-fix)
  - no-wildcard-imports
  - staged/diff secret scan
  - fleet fast guardrails (file size, SPEC/ADR drift, workflow inventory)
  - quality-check (no TODOs/FIXMEs)
  - no-debug-statements
  - no-print-in-src
  - prettier (yaml/json/md)

PRE-PUSH (runs before push, target <3 minutes):
  - mypy (type check)
  - bandit (security scan)
  - pip-audit (bounded Python dependency audit)
  - pytest (unit tests)
  - fleet pre-push guardrails

MANUAL COMMANDS:
  pre-commit run --all-files      # Run all pre-commit hooks
  pre-commit run --hook-stage pre-push  # Run pre-push hooks manually
  make ci-local                   # Optional broad local confidence target
  pre-commit autoupdate           # Update hook versions
""")


def main() -> None:
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("INSTALLING GIT HOOKS")
    logger.info("=" * 60)

    try:
        install_pre_commit()
        install_hooks()
        install_push_hooks()
        install_dev_dependencies()
        install_merge_drivers()
        verify_installation()
        print_summary()
        logger.info("\n[SUCCESS] All hooks installed successfully!")
        logger.info("Your commits will now be checked locally before reaching CI.")

    except subprocess.CalledProcessError as e:
        logger.info(f"\n[ERROR] Command failed: {e}")
        logger.info(f"  stdout: {e.stdout}")
        logger.info(f"  stderr: {e.stderr}")
        sys.exit(1)
    except OSError as e:
        logger.info(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
