"""Installed-artifact contracts for the canonical standalone Sidekick."""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404 - fixed interpreters and local artifacts
import sys
import venv
import zipfile
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WHEEL_MODULES = {
    "shared/python/sidekick/__main__.py",
    "shared/python/sidekick/persistence/__init__.py",
    "shared/python/sidekick/persistence/schema.py",
    "shared/python/sidekick/persistence/state_profile.py",
    "shared/python/sidekick/standalone/__init__.py",
    "shared/python/sidekick/standalone/onboarding.py",
    "shared/python/sidekick/standalone/preferences.py",
    "shared/python/sidekick/standalone/runner.py",
    "shared/python/sidekick/standalone/session_store.py",
    "shared/python/sidekick/standalone/window.py",
}


def _run(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 - fixed interpreter and local paths
        args,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _isolated_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def _venv_python(root: Path) -> Path:
    executable = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    assert executable.is_file(), f"virtualenv interpreter is missing: {executable}"
    return executable


def _venv_script(root: Path, name: str) -> Path:
    """Return an installed console-script path for the current platform."""
    scripts_dir = root / ("Scripts" if os.name == "nt" else "bin")
    suffix = ".exe" if os.name == "nt" else ""
    executable = scripts_dir / f"{name}{suffix}"
    assert executable.is_file(), f"console script is missing: {executable}"
    return executable


def test_built_wheel_contains_and_executes_canonical_standalone_sidekick(
    tmp_path: Path,
) -> None:
    """The exact built wheel must work without source-checkout import leakage."""
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    env = _isolated_environment()
    build_result = _run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(dist_dir),
        ],
        cwd=_REPO_ROOT,
        env=env,
    )
    assert build_result.returncode == 0, build_result.stdout + build_result.stderr
    wheels = list(dist_dir.glob("ud_tools-*.whl"))
    assert len(wheels) == 1, f"expected one exact wheel, found: {wheels}"
    wheel = wheels[0].resolve()

    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
    assert _WHEEL_MODULES <= wheel_names

    venv_root = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_root)
    python = _venv_python(venv_root)
    install_result = _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            str(wheel),
        ],
        cwd=tmp_path,
        env=env,
    )
    assert install_result.returncode == 0, install_result.stdout + install_result.stderr

    help_result = _run(
        [str(python), "-m", "sidekick", "--help"],
        cwd=tmp_path,
        env=env,
        timeout=30,
    )
    assert help_result.returncode == 0, help_result.stdout + help_result.stderr
    assert "Standalone Sidekick launcher and headless dispatcher." in help_result.stdout

    console_help = _run(
        [str(_venv_script(venv_root, "sidekick")), "--help"],
        cwd=tmp_path,
        env=env,
        timeout=30,
    )
    assert console_help.returncode == 0, console_help.stdout + console_help.stderr
    assert "Standalone Sidekick launcher and headless dispatcher." in (
        console_help.stdout
    )

    probe = """
import importlib
import importlib.metadata
import json
from pathlib import Path

names = (
    "sidekick.persistence.schema",
    "shared.python.sidekick.persistence.schema",
    "src.shared.python.sidekick.persistence.schema",
)
modules = [importlib.import_module(name) for name in names]
assert modules[0] is modules[1] is modules[2]
origin = Path(modules[0].__file__).resolve()
requirements = importlib.metadata.requires("ud-tools") or []
assert any(item.lower().startswith("platformdirs>=4.2.0") for item in requirements)
print(json.dumps({"origin": str(origin), "requirements": requirements}))
"""
    probe_result = _run(
        [str(python), "-c", probe],
        cwd=tmp_path,
        env=env,
        timeout=30,
    )
    assert probe_result.returncode == 0, probe_result.stdout + probe_result.stderr
    payload = json.loads(probe_result.stdout)
    origin = Path(payload["origin"]).resolve()
    assert origin.is_relative_to(venv_root.resolve())
    assert not origin.is_relative_to(_REPO_ROOT.resolve())
