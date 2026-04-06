import logging
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def run(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    if result.stdout:
        logger.info("%s", result.stdout.rstrip())
    if result.stderr:
        logger.error("%s", result.stderr.rstrip())

run(["git", "checkout", "main"])
run(["git", "pull", "--rebase"])
run(["git", "checkout", "-b", "feat/chaotic-pendulum-screensaver"])
run(["git", "add", "src/chaotic_pendulum_screensaver", "tests/test_chaotic_pendulum_screensaver"])
run(["git", "commit", "-m", "feat(shared): incorporate chaotic pendulum screensaver\n\nAdheres to TDD, DbC, and DRY."])
run(["git", "push", "-u", "origin", "feat/chaotic-pendulum-screensaver"])
run(["gh", "pr", "create", "--title", "feat: Chaotic Pendulum Screensaver", "--body", "Incorporated chaotic pendulum screensaver into shared Tools.", "--base", "main"])
run(["gh", "pr", "merge", "--auto", "--squash"])
