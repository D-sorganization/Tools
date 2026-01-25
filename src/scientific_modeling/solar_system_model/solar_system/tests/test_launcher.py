"""Tests for the quick-launch helper."""

import os
import sys
import unittest

from pathlib import Path

sys.path.insert(0, Path(Path(os.path.abspath(__file__).parent.parent)))

from launcher import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    build_launch_command,
    check_dependencies,
)


class LauncherTests(unittest.TestCase):
    """Validate helper functions used by the launcher."""

    def test_build_launch_command_defaults(self) -> None:
        """Defaults should create a windowed command with default dimensions."""

        command = build_launch_command()

        self.assertEqual(command[0:3], [sys.executable, "-m", "solar_system.main"])
        self.assertIn(str(DEFAULT_WIDTH), command)
        self.assertIn(str(DEFAULT_HEIGHT), command)
        self.assertNotIn("--fullscreen", command)
        self.assertNotIn("--no-antialiasing", command)

    def test_build_launch_command_with_options(self) -> None:
        """Optional settings should be reflected in the command list."""

        command = build_launch_command(
            width=1024,
            height=768,
            fullscreen=True,
            start_date="2024-01-01",
            enable_antialiasing=False,
        )

        self.assertIn("--fullscreen", command)
        self.assertIn("--no-antialiasing", command)
        self.assertIn("2024-01-01", command)
        self.assertIn("1024", command)
        self.assertIn("768", command)

    def test_dependency_check_reports_missing(self) -> None:
        """Missing modules should be reported with guidance."""

        def fake_spec_finder(name: str) -> object | None:
            return None if name == "pygame" else object()

        status = check_dependencies(spec_finder=fake_spec_finder)

        self.assertFalse(status.ok)
        self.assertIn("pygame", status.missing)
        self.assertIn("pygame", status.guidance)

    def test_dependency_check_success(self) -> None:
        """When all modules are present, status should be OK."""

        status = check_dependencies(spec_finder=lambda name: object())

        self.assertTrue(status.ok)
        self.assertEqual(status.missing, [])
        self.assertEqual(status.guidance, {})


if __name__ == "__main__":
    unittest.main()
