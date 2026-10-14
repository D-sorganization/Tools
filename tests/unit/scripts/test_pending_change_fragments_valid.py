"""Every pending change fragment must pass ``changes_fragment.py validate``.

``collate-changes.yml`` validates all fragments before folding any of them, so
one malformed fragment (no YAML frontmatter, #5609) aborts the collation of
every other merged pull request until someone repairs it by hand. Catch it on
the pull request that adds it instead.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI = REPO_ROOT / "shared_scripts" / "changes_fragment.py"


def test_all_pending_fragments_validate() -> None:
    result = subprocess.run(
        [sys.executable, str(CLI), "validate", "--repo-root", str(REPO_ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "A pending fragment would abort collate-changes.yml:\n"
        + result.stdout
        + result.stderr
    )
