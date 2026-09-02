#!/usr/bin/env python3
"""Pre-commit hook: fix up a stale module inventory while a merge is landing.

Why this exists (and why the merge driver alone isn't enough)
---------------------------------------------------------------
``scripts/git/module_inventory_merge_driver.py`` stops
``manuals/tools/manifests/module-inventory.json`` and its shards from
blocking a ``git merge`` with conflict markers. But empirically (verified
locally while building #4818), a git merge driver is invoked *while* the
merge is still being computed -- before git has finished writing every
other path's merged content to the working tree. A driver that tries to
regenerate the inventory at that point can only see a partial merge (e.g.
its own branch's tree plus whatever git happened to have materialized so
far), so its result can legitimately still be stale once the merge
actually finishes.

By the time git is about to create the merge commit, that limitation is
gone: the *whole* merged tree is checked out on disk, which is exactly what
``scripts/build_tools_module_inventory.py`` needs to produce a correct
result (it scans ``git ls-files`` plus on-disk content, nothing else). This
hook runs at that moment -- the ``pre-commit`` git hook stage, i.e. right
before a commit (including a merge commit) is created -- detects whether a
merge is in progress, and if the inventory is stale, regenerates and
re-stages it so the merge commit that's about to be created already carries
a fresh, correct inventory. On an ordinary (non-merge) commit this is a
fast no-op: existing pre-commit hooks (``tools-module-inventory-freshness``)
still enforce that contributors regenerate and review the diff themselves,
unchanged.

Registered like any other local pre-commit hook (see
``.pre-commit-config.yaml``); no separate installation step beyond the
repo's existing ``pre-commit install``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REGEN_MODULE = "scripts.build_tools_module_inventory"


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _merge_in_progress(root: Path) -> bool:
    """Return whether a ``git merge`` (or merge-like ``git pull``) is landing.

    ``MERGE_HEAD`` exists from the moment conflict resolution finishes
    until the merge commit is created, which is exactly this hook's
    window.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--git-path", "MERGE_HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return (root / result.stdout.strip()).is_file()


def fixup(root: Path) -> int:
    """Regenerate and re-stage the inventory if it is stale mid-merge.

    Returns a process exit code: 0 to let the commit proceed, non-zero to
    block it (regeneration itself failed, which is a real problem worth
    surfacing rather than silently committing broken output).
    """
    if not _merge_in_progress(root):
        return 0

    check = subprocess.run(
        [sys.executable, "-m", _REGEN_MODULE, "--check"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        return 0

    regenerate = subprocess.run(
        [sys.executable, "-m", _REGEN_MODULE],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if regenerate.returncode != 0:
        sys.stderr.write(
            "regenerate-module-inventory-during-merge: regeneration failed\n"
        )
        sys.stderr.write(regenerate.stdout)
        sys.stderr.write(regenerate.stderr)
        return regenerate.returncode

    add = subprocess.run(
        [
            "git",
            "add",
            "--",
            "manuals/tools/manifests/module-inventory.json",
            "manuals/tools/manifests/module-inventory/",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if add.returncode != 0:
        sys.stderr.write(
            "regenerate-module-inventory-during-merge: failed to stage the "
            "regenerated inventory\n"
        )
        sys.stderr.write(add.stdout)
        sys.stderr.write(add.stderr)
        return add.returncode

    sys.stderr.write(
        "regenerate-module-inventory-during-merge: inventory was stale "
        "after the merge; regenerated and staged it for the merge commit\n"
    )
    return 0


def main() -> int:
    return fixup(_repo_root())


if __name__ == "__main__":
    raise SystemExit(main())
