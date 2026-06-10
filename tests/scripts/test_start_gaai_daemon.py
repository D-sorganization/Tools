"""Regression tests for the GAAI daemon launcher safety settings (#3291)."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "start-gaai-daemon.sh"


def _bash_executable() -> str:
    """Return a usable Bash executable for running shell-script regressions."""
    if sys.platform == "win32":
        for candidate in (
            Path(r"C:\Program Files\Git\bin\bash.exe"),
            Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
        ):
            if candidate.exists():
                return str(candidate)

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required for start-gaai-daemon.sh regression tests")
    return bash


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _bash_path(path: Path) -> str:
    if sys.platform != "win32":
        return str(path)

    drive = path.drive.rstrip(":").lower()
    relative_path = path.relative_to(path.anchor).as_posix()
    return f"/{drive}/{relative_path}"


def _copy_launcher_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    daemon_dir = repo / ".gaai" / "core" / "scripts"
    daemon_dir.mkdir(parents=True)
    shutil.copy2(_SCRIPT, repo / "start-gaai-daemon.sh")
    _write_executable(
        daemon_dir / "delivery-daemon.sh",
        "#!/usr/bin/env bash\nprintf 'daemon %s\\n' \"$*\"\n",
    )
    _init_local_git_remote(repo, tmp_path / "origin.git")
    return repo / "start-gaai-daemon.sh"


def _run_git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_local_git_remote(repo: Path, remote: Path) -> None:
    """Create a local staging branch and origin for the launcher's git checks."""
    _run_git(["init", "-b", "staging"], repo)
    _run_git(["config", "user.email", "test@example.invalid"], repo)
    _run_git(["config", "user.name", "Test User"], repo)
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    _run_git(["add", "README.md"], repo)
    _run_git(["commit", "-m", "test fixture"], repo)
    _run_git(["init", "--bare", str(remote)], repo)
    _run_git(["remote", "add", "origin", str(remote)], repo)
    _run_git(["push", "-u", "origin", "staging"], repo)


def _stub_dependencies(bin_dir: Path) -> None:
    bin_dir.mkdir()
    _write_executable(bin_dir / "tmux", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(bin_dir / "claude", "#!/usr/bin/env bash\nexit 0\n")


def _launcher_env(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    home_dir = tmp_path / "home"
    _stub_dependencies(bin_dir)
    home_dir.mkdir()

    env = os.environ.copy()
    env["HOME"] = str(home_dir)
    path_separator = ":" if sys.platform == "win32" else os.pathsep
    env["PATH"] = path_separator.join((_bash_path(bin_dir), env.get("PATH", "")))
    return env


def test_dry_run_preserves_existing_global_claude_settings(tmp_path: Path) -> None:
    """Dry-run must not clobber settings or disable safety prompts globally."""
    script = _copy_launcher_fixture(tmp_path)
    env = _launcher_env(tmp_path)
    settings_path = Path(env["HOME"]) / ".claude" / "settings.json"
    settings_path.parent.mkdir()
    original_settings = {
        "permissions": {"allow": ["Bash(git status:*)"]},
        "hooks": {"Stop": [{"matcher": "*", "hooks": []}]},
        "model": "claude-sonnet-4-5",
    }
    settings_path.write_text(json.dumps(original_settings), encoding="utf-8")

    result = subprocess.run(
        [_bash_executable(), str(script), "--dry-run"],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings == original_settings
    assert "skipDangerousModePermissionPrompt" not in settings
