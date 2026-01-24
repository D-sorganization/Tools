#!/usr/bin/env python3
import logging

# Use shared logging utility
try:
    from utils.logging_utils import init_default_logging
except ImportError:
    # Fallback
    def init_default_logging():
        init_default_logging()
import shutil
import subprocess

# Use shared subprocess utility
try:
    from utils.subprocess_utils import run_command
except ImportError:
    # Fallback
    import subprocess
    run_command = subprocess.run
import sys

init_default_logging()
logger = logging.getLogger("VideoPlatformLauncher")


def main() -> None:
    logger.info("Starting Video Processor Platform...")

    # Check for npm/pnpm
    pnpm = shutil.which("pnpm")
    npm = shutil.which("npm")

    cmd = []
    if pnpm:
        cmd = [pnpm, "run", "dev"]
    elif npm:
        cmd = [npm, "run", "dev"]
    else:
        logger.error("Node.js package manager (npm or pnpm) not found!")
        sys.exit(1)

    try:
        # Run in the current directory
        run_command(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Stopped.")
    except Exception as e:
        logger.error(f"Failed to run platform: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
