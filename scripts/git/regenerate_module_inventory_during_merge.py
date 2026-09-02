#!/usr/bin/env python3
"""``pre-merge-commit`` hook: fix up a stale module inventory after a merge.

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
script runs at exactly that moment via git's ``pre-merge-commit`` hook --
note this is a *distinct* hook from plain ``pre-commit``, which git does
NOT invoke for merge commits at all (confirmed empirically: a raw
``.git/hooks/pre-commit`` script produced zero output across a real
``git merge``, while a raw ``.git/hooks/pre-merge-commit`` script fired
every time, with the full merged tree already on disk). If the inventory
is stale, this regenerates and re-stages it so the merge commit that's
about to be created already carries a fresh, correct inventory.

This script does *not* check ``MERGE_HEAD`` to confirm a merge is
happening: empirically (confirmed with a raw, framework-free hook script),
``MERGE_HEAD`` does not exist yet at the point ``pre-merge-commit`` fires
-- git writes it later, as part of actually creating the commit. That
guard would always read false and silently no-op the whole hook. It isn't
needed anyway: this script is only ever invoked by the ``pre-merge-commit``
git hook, which by definition only fires while a merge commit is being
created.

Registered as a local pre-commit-framework hook (see
``.pre-commit-config.yaml``, ``stages: [pre-merge-commit]``), which needs
its own install step beyond the default ``pre-commit install`` --
``pre-commit install --hook-type pre-merge-commit`` -- already wired into
``scripts/setup_precommit.sh`` and ``scripts/setup_hooks.py``.
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


def fixup(root: Path) -> int:
    """Regenerate and re-stage the inventory if it is stale.

    Returns a process exit code: 0 to let the commit proceed, non-zero to
    block it (regeneration itself failed, which is a real problem worth
    surfacing rather than silently committing broken output).
    """
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
