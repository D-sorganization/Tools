"""Fail-closed setuptools hooks for optional Rate of Closure web assets."""

from __future__ import annotations

import os
import re
import shutil

# Subprocesses below use fixed executables/argv and never invoke a shell.
import subprocess  # nosec B404
import sys
from pathlib import Path

from setuptools.command.build_py import build_py
from setuptools.errors import SetupError

_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class RateWebBuildPy(build_py):
    """Admit web package data only when it matches the current release source."""

    def run(self) -> None:
        root = Path(__file__).resolve().parent
        web_root = root / "src" / "rate_of_closure" / "web" / "dist"
        revision = os.environ.get("ROC_RELEASE_REVISION")
        if web_root.exists():
            self._verify_web_distribution(root, revision)
        elif revision is not None:
            raise SetupError("ROC_RELEASE_REVISION requires a built web distribution")
        super().run()

    @staticmethod
    def _verify_web_distribution(root: Path, revision: str | None) -> None:
        if revision is None or _COMMIT.fullmatch(revision) is None:
            raise SetupError("web package data requires an exact ROC_RELEASE_REVISION")
        if not (root / ".git").exists():
            raise SetupError("qualified web package data requires a Git checkout")
        git = shutil.which("git")
        if git is None:
            raise SetupError("git is required to verify web release identity")
        # The executable is resolved absolutely and argv is constant.
        result = subprocess.run(  # nosec B603
            [git, "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout.strip() != revision:
            raise SetupError("ROC_RELEASE_REVISION does not match checkout HEAD")
        status = subprocess.run(  # nosec B603
            [git, "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout:
            raise SetupError("web distribution source checkout must be clean")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(root / "src")
        # The only variable argument already passed the exact commit regex.
        subprocess.run(  # nosec B603
            [
                sys.executable,
                "-m",
                "rate_of_closure.web_distribution.verify_install",
                "--expected-revision",
                revision,
            ],
            cwd=root,
            env=environment,
            check=True,
        )
