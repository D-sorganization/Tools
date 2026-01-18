#!/usr/bin/env python3
import logging
import shutil
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
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
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Stopped.")
    except Exception as e:
        logger.error(f"Failed to run platform: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
