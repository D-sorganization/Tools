"""Quick-launch helpers for the Solar System Simulation.

This module provides a lightweight launcher that can be double-clicked or
invoked from the command line to start the simulation with sensible defaults.
It performs dependency checks before spawning the main application so users get
clear guidance instead of obscure import errors.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib.util import find_spec

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


@dataclass
class DependencyStatus:
    """Represents the result of dependency checks."""

    ok: bool
    missing: list[str]
    guidance: dict[str, str]


def _has_module(name: str, spec_finder: Callable[[str], object] = find_spec) -> bool:
    """Return True if a module can be imported.

    The optional ``spec_finder`` argument makes the function easy to test by
    injecting a deterministic stub.

    Note: This function is kept for backward compatibility and testing.
    For new code, use utils.dependency_checker.has_module instead.
    """
    return spec_finder(name) is not None


def check_dependencies(
    spec_finder: Callable[[str], object] = find_spec,
) -> DependencyStatus:
    """Check whether required visualization dependencies are available.

    Uses shared dependency checker utility when available, with fallback
    to local implementation for backward compatibility.
    """
    try:
        # Try to use shared utility
        import sys

        # Add utils to path
        try:
            from utils.path_helpers import ensure_utils_in_path

            ensure_utils_in_path()
        except ImportError:
            # Fallback
            from utils.path_helpers import get_project_root_from_file

            repo_root = get_project_root_from_file(__file__)
            sys.path.insert(0, str(repo_root / "src" / "python" / "src"))

        from utils.dependency_checker import DependencyStatus
        from utils.dependency_checker import check_dependencies as check_deps

        required = {
            "numpy": "pip install numpy",
            "pygame": "pip install pygame",
            "OpenGL": "pip install PyOpenGL PyOpenGL_accelerate",
        }

        return check_deps(required, spec_finder=spec_finder)
    except ImportError:
        # Fallback to local implementation
        required = {
            "numpy": "pip install numpy",
            "pygame": "pip install pygame",
            "OpenGL": "pip install PyOpenGL PyOpenGL_accelerate",
        }

        missing = [name for name in required if not _has_module(name, spec_finder)]

        return DependencyStatus(
            ok=not missing,
            missing=missing,
            guidance={name: required[name] for name in missing},
        )


def build_launch_command(
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fullscreen: bool = False,
    start_date: str | None = None,
    enable_antialiasing: bool = True,
) -> list[str]:
    """Compose the command that will start the simulation."""

    command = [
        sys.executable,
        "-m",
        "solar_system.main",
        "--width",
        str(width),
        "--height",
        str(height),
    ]

    if fullscreen:
        command.append("--fullscreen")

    if not enable_antialiasing:
        command.append("--no-antialiasing")

    if start_date:
        command.extend(["--start-date", start_date])

    return command


def launch_quickstart(
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fullscreen: bool = False,
    start_date: str | None = None,
    enable_antialiasing: bool = True,
) -> int:
    """Launch the simulation with helpful dependency checks.

    Returns an exit code suitable for ``sys.exit``.
    """

    status = check_dependencies()
    if not status.ok:
        print("Missing dependencies detected:")
        for name in status.missing:
            guidance = status.guidance.get(name)
            hint = f" ({guidance})" if guidance else ""
            print(f"  - {name}{hint}")
        print(
            "\nInstall the missing packages and try again. "
            "Recommended: pip install -r solar_system/requirements.txt"
        )
        return 1

    command = build_launch_command(
        width=width,
        height=height,
        fullscreen=fullscreen,
        start_date=start_date,
        enable_antialiasing=enable_antialiasing,
    )

    print("Starting Solar System Simulation...")
    print(f"  Resolution: {width}x{height}")
    if fullscreen:
        print("  Mode: fullscreen")
    else:
        print("  Mode: windowed (toggle fullscreen via OS controls)")
    print("  Tip: Use H to toggle on-screen help once the window opens.\n")

    return subprocess.call(command)


__all__ = [
    "DependencyStatus",
    "check_dependencies",
    "build_launch_command",
    "launch_quickstart",
    "DEFAULT_WIDTH",
    "DEFAULT_HEIGHT",
]
