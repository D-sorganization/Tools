"""Protect the test suite from verification-test Playwright stubs."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VERIFICATION_TEST = (
    REPOSITORY_ROOT / "tests" / "verification" / "test_verification_lod.py"
)


def test_importing_verification_lod_test_does_not_register_playwright_stub() -> None:
    """The verification tests must not leave fake Playwright modules in pytest."""
    script = textwrap.dedent(
        """
        import importlib.util
        import pathlib
        import sys

        test_path = pathlib.Path(sys.argv[1])
        assert "playwright" not in sys.modules
        playwright_installed = importlib.util.find_spec("playwright") is not None
        modules_before = {
            name: module
            for name, module in sys.modules.items()
            if name == "playwright" or name.startswith("playwright.")
        }
        spec = importlib.util.spec_from_file_location(
            "isolated_verification_lod", test_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules_after = {
            name: module
            for name, module in sys.modules.items()
            if name == "playwright" or name.startswith("playwright.")
        }
        assert modules_after == modules_before, (
            "verification test leaked Playwright stubs"
        )
        if playwright_installed:
            from playwright.sync_api import Browser

            assert Browser.__name__ == "Browser"
        """
    )
    child_env = os.environ.copy()
    child_env.pop("GH_TOKEN", None)
    child_env.pop("GITHUB_TOKEN", None)

    result = subprocess.run(
        [sys.executable, "-c", script, str(VERIFICATION_TEST)],
        cwd=REPOSITORY_ROOT,
        env=child_env,
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
