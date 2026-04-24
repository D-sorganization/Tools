import logging
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def run(cmd: list[str]) -> None:
    """Execute a shell command with logging and error handling.

    Pre: cmd is a non-empty list of strings.
    Post: Logs output and errors; raises CalledProcessError on failure.
    """
    if not cmd:
        raise ValueError("cmd must be non-empty")
    try:
        logger.info("Running: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=REPO_ROOT, check=True
        )
        if result.stdout:
            logger.info("%s", result.stdout.rstrip())
        if result.stderr:
            logger.warning("%s", result.stderr.rstrip())
    except subprocess.CalledProcessError as e:
        logger.error("Command failed with exit code %d", e.returncode)
        if e.stdout:
            logger.error("stdout: %s", e.stdout.rstrip())
        if e.stderr:
            logger.error("stderr: %s", e.stderr.rstrip())
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(["git", "checkout", "main"])
    run(["git", "pull", "--rebase"])
    run(["git", "checkout", "-b", "feat/chaotic-pendulum-screensaver"])
    run(
        [
            "git",
            "add",
            "src/chaotic_pendulum_screensaver",
            "tests/test_chaotic_pendulum_screensaver",
        ]
    )
    run(
        [
            "git",
            "commit",
            "-m",
            "feat(shared): incorporate chaotic pendulum screensaver\n\nAdheres to TDD, DbC, and DRY.",
        ]
    )
    run(["git", "push", "-u", "origin", "feat/chaotic-pendulum-screensaver"])
    run(
        [
            "gh",
            "pr",
            "create",
            "--title",
            "feat: Chaotic Pendulum Screensaver",
            "--body",
            "Incorporated chaotic pendulum screensaver into shared Tools.",
            "--base",
            "main",
        ]
    )
    run(["gh", "pr", "merge", "--auto", "--squash"])
