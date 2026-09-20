"""Build a standalone executable of the Rate of Closure Impact Explorer.

Wraps PyInstaller with the right entry point and options so users can
produce a double-clickable program and experiment without a Python
environment:

    python src/rate_of_closure/build_executable.py

The result lands in ``dist/RateOfClosureExplorer`` (one-folder mode:
faster startup and easier antivirus review than one-file). Requires
``pip install pyinstaller`` in the environment that already runs the
tool. The shareable web equivalent is ``npm run build`` inside
``src/rate_of_closure/web`` (outputs to static bundle).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from rate_of_closure.packaging.build_artifact import build_frozen_artifact

APP_NAME = "RateOfClosureExplorer"
_HERE = Path(__file__).resolve().parent


def build(one_file: bool = False) -> Path:
    """Run PyInstaller and return the path to the built executable.

    Args:
        one_file: Retained for backward compatibility. Note: production
            qualification standardizes on one-folder (onedir) mode.

    Returns:
        Path to the produced executable.

    Raises:
        RuntimeError: If PyInstaller is not installed or the build fails.
    """
    dist = _HERE.parent.parent / "dist"
    return Path(build_frozen_artifact(dist_path=dist))


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
