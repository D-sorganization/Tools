"""Guard against developer-specific absolute paths in adapters and scripts.

Hardcoded home directories (e.g. ``C:\\Users\\diete`` or ``/home/dieterolson``)
resolve only on the original developer's machine. On any other user, host, or CI
runner they are dead paths: AI CLI adapters fall through to a non-existent
fallback ("CLI not found"), and fleet scripts fail with ``FileNotFoundError``
before doing useful work. This test makes such literals a hard failure.

See Tools issue #3278.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Repo root: tests/scripts/<this file> -> parents[2] is the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Directories scanned for developer-specific path literals.
_SCANNED_DIRS = (
    _REPO_ROOT / "src" / "shared" / "python" / "ai" / "adapters",
    _REPO_ROOT / "scripts",
)

# Windows ``C:\Users\<name>`` and POSIX ``/home/<name>/`` home-directory roots.
# A bare username token (letters, digits, dot, underscore, hyphen) follows the
# separator; this test file itself only contains the patterns inside raw
# strings/comments, which are excluded below.
_HARDCODED_HOME_RE = re.compile(
    r"[A-Za-z]:\\Users\\[A-Za-z0-9._-]+" r"|/home/[A-Za-z0-9._-]+/",
)


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for base in _SCANNED_DIRS:
        if not base.exists():
            continue
        files.extend(sorted(base.rglob("*.py")))
    return files


@pytest.mark.unit
def test_no_developer_home_paths() -> None:
    """No source file may embed a developer-specific home directory."""
    offenders: dict[str, list[str]] = {}
    self_path = Path(__file__).resolve()
    for path in _iter_python_files():
        if path.resolve() == self_path:
            # This guard file legitimately documents the patterns it forbids.
            continue
        text = path.read_text(encoding="utf-8")
        matches = _HARDCODED_HOME_RE.findall(text)
        if matches:
            offenders[str(path.relative_to(_REPO_ROOT))] = sorted(set(matches))

    assert not offenders, (
        "Developer-specific hardcoded home paths found (use shutil.which / "
        "Path.home() / env vars instead):\n"
        + "\n".join(f"  {f}: {paths}" for f, paths in offenders.items())
    )
