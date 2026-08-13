"""Build a standalone executable of the Rate of Closure Impact Explorer.

Wraps PyInstaller with the right entry point and options so users can
produce a double-clickable program and experiment without a Python
environment:

    python src/rate_of_closure/build_executable.py

The result lands in ``dist/RateOfClosureExplorer`` (one-folder mode:
faster startup and easier antivirus review than one-file). Requires
``pip install pyinstaller`` in the environment that already runs the
tool. The supported shareable web release is the static Vite bundle built
with ``npm ci`` and ``npm run build`` inside ``src/rate_of_closure/web``.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess  # nosec B404  # Fixed argument list; no user input.
import sys
from pathlib import Path

APP_NAME = "RateOfClosureExplorer"
_HERE = Path(__file__).resolve().parent


def build(one_file: bool = False) -> Path:
    """Run PyInstaller and return the path to the built executable.

    Args:
        one_file: Bundle into a single self-extracting executable
            instead of the default one-folder layout.

    Returns:
        Path to the produced executable.

    Raises:
        RuntimeError: If PyInstaller is not installed or the build fails.
    """
    if importlib.util.find_spec("PyInstaller") is None:
        raise RuntimeError(
            "PyInstaller is not installed - run: pip install pyinstaller"
        )
    dist = _HERE.parent.parent / "dist"
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--windowed",
        "--name",
        APP_NAME,
        "--onefile" if one_file else "--onedir",
        "--distpath",
        str(dist),
        str(_HERE / "launch_pyqt6.py"),
    ]
    completed = subprocess.run(command, check=False)  # nosec B603  # Fixed argv.
    if completed.returncode != 0:
        raise RuntimeError(f"PyInstaller failed with code {completed.returncode}")

    suffix = ".exe" if sys.platform == "win32" else ""
    built = (
        dist / f"{APP_NAME}{suffix}"
        if one_file
        else dist / APP_NAME / f"{APP_NAME}{suffix}"
    )
    if not built.exists():
        raise RuntimeError(f"expected executable missing: {built}")
    return built


def main() -> int:
    """CLI entry point."""
    one_file = "--onefile" in sys.argv[1:]
    built = build(one_file=one_file)
    sys.stdout.write(f"Built: {built}\n")
    if shutil.which("explorer") and sys.platform == "win32":
        sys.stdout.write("Open the dist folder to run or share it.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
