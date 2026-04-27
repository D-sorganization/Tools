"""
Shared subprocess execution utility for consistent process management.

This module provides reusable functions for executing subprocess commands
across the repository, following DRY principles.
"""

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_command(
    command: list[str],
    cwd: Path | str | None = None,
    timeout: int | None = None,
    capture_output: bool = True,
    check: bool = False,
    text: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command with consistent error handling.

    Args:
        command: Command to run as list of strings
        cwd: Working directory for command
        timeout: Timeout in seconds
        capture_output: Capture stdout and stderr
        check: Raise exception on non-zero exit
        text: Return output as text (not bytes)
        env: Environment variables

    Returns:
        CompletedProcess object

    Raises:
        subprocess.CalledProcessError: If check=True and command fails
        subprocess.TimeoutExpired: If command times out
    """
    cwd_path = Path(cwd) if cwd else None
    if cwd_path and not cwd_path.exists():
        raise FileNotFoundError(f"Working directory not found: {cwd_path}")

    logger.debug(f"Running command: {' '.join(command)}")
    if cwd_path:
        logger.debug(f"Working directory: {cwd_path}")

    try:
        result = subprocess.run(
            command,
            cwd=str(cwd_path) if cwd_path else None,
            timeout=timeout,
            capture_output=capture_output,
            check=check,
            text=text,
            env=env,
        )
        if result.returncode == 0:
            logger.debug("Command completed successfully")
        else:
            logger.warning(f"Command exited with code {result.returncode}")
        return result
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout}s: {' '.join(command)}")
        raise
    except (subprocess.SubprocessError, OSError, FileNotFoundError) as e:
        logger.error(f"Unexpected error running command: {e}")
        raise


def run_python_script(
    script_path: Path | str,
    args: list[str] | None = None,
    cwd: Path | str | None = None,
    timeout: int | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a Python script with consistent error handling.

    Args:
        script_path: Path to Python script
        args: Additional arguments to pass to script
        cwd: Working directory
        timeout: Timeout in seconds
        check: Raise exception on non-zero exit

    Returns:
        CompletedProcess object
    """
    script = Path(script_path)
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")

    command = [sys.executable, str(script)]
    if args:
        command.extend(args)

    return run_command(command, cwd=cwd, timeout=timeout, check=check)


def run_pip_command(
    command: str,
    packages: list[str] | None = None,
    requirements_file: Path | str | None = None,
    upgrade: bool = False,
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a pip command with consistent error handling.

    Args:
        command: Pip command (install, uninstall, etc.)
        packages: List of packages (for install/uninstall)
        requirements_file: Path to requirements.txt file
        upgrade: Upgrade packages if already installed
        timeout: Timeout in seconds
        check: Raise exception on non-zero exit

    Returns:
        CompletedProcess object
    """
    if not (command is not None):
        raise ValueError("command must be provided")
    pip_cmd = [sys.executable, "-m", "pip", command]

    if upgrade:
        pip_cmd.append("--upgrade")

    if command == "install":
        if requirements_file:
            pip_cmd.extend(["-r", str(requirements_file)])
        elif packages:
            pip_cmd.extend(packages)
        else:
            raise ValueError("Must provide either packages or requirements_file")
    elif packages:
        pip_cmd.extend(packages)

    return run_command(pip_cmd, timeout=timeout, check=check)


def check_command_available(command: str) -> bool:
    """Check if a command is available in PATH.

    Args:
        command: Command name to check

    Returns:
        True if command is available, False otherwise
    """
    try:
        result = run_command(
            ["which", command] if sys.platform != "win32" else ["where", command],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
    except (KeyError, ValueError, TypeError):
        return False


def get_command_output(
    command: list[str],
    cwd: Path | str | None = None,
    timeout: int | None = None,
) -> str:
    """Get stdout output from a command.

    Args:
        command: Command to run
        cwd: Working directory
        timeout: Timeout in seconds

    Returns:
        Command stdout as string
    """
    if not (command is not None):
        raise ValueError("command must be provided")
    result = run_command(command, cwd=cwd, timeout=timeout, check=True)
    return result.stdout.strip()
