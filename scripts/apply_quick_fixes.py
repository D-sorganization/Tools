#!/usr/bin/env python3
"""
Apply safe, automated quick fixes to the codebase.
"""

import logging
import subprocess
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent.resolve()


def run_command(command: list[str], description: str):
    logger.info(f"Running: {description}...")
    try:
        result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Successfully ran {description}.")
        else:
            logger.warning(f"Issues found during {description}:\n{result.stderr}")
            # Some tools return non-zero on changes or warnings, so we don't necessarily fail
    except Exception as e:
        logger.error(f"Failed to run {description}: {e}")


def main():
    logger.info("Starting Quick Fixes...")

    # 1. Ruff: Sort imports (I), Upgrade syntax (UP), Remove unused imports (F401)
    # We also check for other safe fixes if available by default
    run_command(
        ["ruff", "check", "--fix", "--select", "I,UP,F401", "."],
        "Ruff (Imports, Upgrades, Unused)",
    )

    # 2. Black: Format code
    run_command(["black", "."], "Black (Formatting)")

    logger.info("Quick fixes applied.")


if __name__ == "__main__":
    main()
