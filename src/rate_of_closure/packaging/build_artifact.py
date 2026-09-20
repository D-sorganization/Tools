"""Build the standalone one-folder PyQt6 executable using PyInstaller.

Wraps PyInstaller with the explicit rate_of_closure.spec specification file,
ensuring all required assets, models, and dependencies are bundled without
relying on repository _bootstrap.py or dynamic registration.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

APP_NAME = "RateOfClosureExplorer"
_PACKAGING_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGING_DIR.parents[2]


def build_frozen_artifact(
    dist_path: Path | None = None,
    *,
    clean: bool = False,
) -> Path:
    """Run PyInstaller with rate_of_closure.spec and return path to built exe.

    Args:
        dist_path: Custom directory for output (defaults to repo root / dist).
        clean: Whether to clean PyInstaller cache before building.

    Returns:
        Path to the produced executable.

    Raises:
        RuntimeError: If PyInstaller is not installed or build fails.
    """
    if importlib.util.find_spec("PyInstaller") is None:
        raise RuntimeError(
            "PyInstaller is not installed - run: pip install pyinstaller"
        )

    out_dist = dist_path if dist_path is not None else _REPO_ROOT / "dist"
    spec_path = _PACKAGING_DIR / "rate_of_closure.spec"
    if not spec_path.is_file():
        raise RuntimeError(f"PyInstaller spec file missing: {spec_path}")

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--distpath",
        str(out_dist),
    ]
    if clean:
        command.append("--clean")
    command.append(str(spec_path))

    env = dict(os.environ)
    src_dir = str(_REPO_ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_dir}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else src_dir
    )

    completed = subprocess.run(command, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"PyInstaller build failed with exit code {completed.returncode}"
        )

    suffix = ".exe" if sys.platform == "win32" else ""
    built_exe = out_dist / APP_NAME / f"{APP_NAME}{suffix}"
    if not built_exe.is_file():
        raise RuntimeError(f"Expected built executable missing at: {built_exe}")

    return built_exe


def main() -> int:
    """CLI entry point for artifact build."""
    try:
        built = build_frozen_artifact()
        sys.stdout.write(f"Successfully built: {built}\n")
        return 0
    except Exception as exc:
        sys.stderr.write(f"Build failed: {exc}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
