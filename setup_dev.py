#!/usr/bin/env python3
import logging
import shutil
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ANSI Color Codes
CYAN = "\033[1;34m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
RED = "\033[1;31m"
RESET = "\033[0m"


def log_step(message: str) -> None:
    """Log a setup step with a coloured [SETUP] prefix.

    Args:
        message: The step description to log.
    """
    logger.info(f"\n{CYAN}[SETUP] {message}{RESET}")


def check_python() -> None:
    """Log the current Python version to verify the environment."""
    log_step("Checking Python environment...")
    logger.info(f"Python version: {sys.version}")


def install_python_deps() -> None:
    """Upgrade pip and install all Python dependencies from requirements.txt."""
    log_step("Installing Python dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    )


def install_node_deps() -> None:
    """Install Node.js dependencies via pnpm, falling back to npm if pnpm is absent."""
    log_step("Installing Node.js dependencies...")

    # Check for pnpm
    pnpm_path = shutil.which("pnpm")
    if not pnpm_path:
        logger.warning(
            f"{YELLOW}Warning: 'pnpm' not found. Attempting to install via npm...{RESET}"
        )
        npm_path = shutil.which("npm")
        if not npm_path:
            logger.error(
                f"{RED}Error: neither 'pnpm' nor 'npm' found. Node.js dependencies skipped.{RESET}"
            )
            return
        try:
            subprocess.check_call(["npm", "install", "-g", "pnpm"])
        except subprocess.CalledProcessError:
            logger.error(
                f"{RED}Error: Failed to install pnpm globally. Please install it manually.{RESET}"
            )
            return

    unit_converter_path = Path("src/web_applications/unit_converter")
    if unit_converter_path.exists():
        logger.info(f"Installing dependencies in {unit_converter_path}...")
        try:
            subprocess.check_call(["pnpm", "install"], cwd=unit_converter_path)
        except subprocess.CalledProcessError:
            logger.error(
                f"{RED}Error: Failed to install dependencies in unit_converter.{RESET}"
            )
    else:
        logger.warning(f"Path not found: {unit_converter_path}")


def main() -> None:
    """Run the full dev setup: check Python, install Python deps, install Node deps."""
    try:
        check_python()
        install_python_deps()
        install_node_deps()
        log_step("Setup complete! You are ready to go.")
    except Exception as e:  # noqa: BLE001
        logger.error(f"\n{RED}Setup failed: {e}{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
