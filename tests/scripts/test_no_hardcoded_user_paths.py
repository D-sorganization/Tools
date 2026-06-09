"""Lint test: no developer-specific absolute paths in adapters or scripts (#3278).

Hardcoded developer home directories (e.g. ``C:\\Users\\diete\\...`` or
``/home/dieterolson/...``) resolve only on the original developer's machine.
In the AI CLI adapters they make the PATH fallbacks dead on every other host;
in the fleet scripts they make every path operation fail with
``FileNotFoundError`` on CI runners or any other checkout. This test fails if
any such literal reappears.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Directories whose source must stay free of developer-specific home paths.
_SCANNED_DIRS = (
    _REPO_ROOT / "src" / "shared" / "python" / "ai" / "adapters",
    _REPO_ROOT / "scripts",
)

# Windows ``C:\Users\<name>`` and POSIX ``/home/<name>/`` developer home paths.
# A backslash or forward slash before ``Users``/``home`` anchors the match so
# unrelated tokens are not flagged.
_DEV_HOME_RE = re.compile(
    r"(?:[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+)|(?:/home/[A-Za-z0-9_.-]+/)"
)


def _python_files() -> list[Path]:
    files: list[Path] = []
    for directory in _SCANNED_DIRS:
        if directory.is_dir():
            files.extend(sorted(directory.rglob("*.py")))
    return files


def test_scanned_dirs_exist() -> None:
    for directory in _SCANNED_DIRS:
        assert directory.is_dir(), f"expected scan directory missing: {directory}"


@pytest.mark.parametrize(
    "path",
    _python_files(),
    ids=lambda p: str(p.relative_to(_REPO_ROOT)),
)
def test_no_developer_home_paths(path: Path) -> None:
    """No file under the scanned dirs may embed a developer-specific home path."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = _DEV_HOME_RE.findall(text)
    assert not matches, (
        f"{path.relative_to(_REPO_ROOT)} contains hardcoded developer "
        f"home path(s): {matches}. Resolve binaries via shutil.which + "
        f"home-relative fallbacks, and derive roots from __file__ / env / CLI."
    )
