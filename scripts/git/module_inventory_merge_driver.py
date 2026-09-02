#!/usr/bin/env python3
"""Git merge driver that regenerates the Tools module inventory.

``manuals/tools/manifests/module-inventory.json`` and its shards under
``manuals/tools/manifests/module-inventory/entries-*.json`` are generated
artifacts (see ``scripts/build_tools_module_inventory.py``): every field,
including the per-shard content hashes, is a pure function of the *other*
tracked files in the working tree. The generator never reads its own prior
output, so a textual conflict on these paths carries no information worth
preserving from either side of a merge -- the correct resolution is always
to throw both versions away and regenerate from whatever the merge already
produced for everything else, exactly what a human resolves the conflict to
by hand today (``git checkout --theirs`` + regenerate + verify).

This script is invoked by git as a custom merge driver, per the contract in
``git help gitattributes`` ("Defining a custom merge driver") and
``git help merge``::

    <driver-command> %O %A %B %L %P

- ``%O`` -- ancestor version (ignored: see rationale above)
- ``%A`` -- current/"ours" version; MUST be overwritten in place with the
  resolved content
- ``%B`` -- other/"theirs" version (ignored)
- ``%L`` -- conflict marker size (unused)
- ``%P`` -- the path, relative to the repository root, the result will be
  stored at

Exit 0 for a clean resolve (git uses whatever is now in %A); exit non-zero
to fall back to git's normal conflicted merge for that path.

The driver command itself is registered as *local* git config by
``scripts/git/install_merge_drivers.py`` -- see that script's docstring for
why ``.gitattributes`` alone cannot do this.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REGEN_MODULE = "scripts.build_tools_module_inventory"


def _repo_root() -> Path:
    """Return the top-level working directory git invoked this driver from.

    Git documents that merge drivers run with the toplevel of the working
    tree as their current directory, but we re-derive it defensively rather
    than assume ``Path.cwd()`` in case a caller (e.g. a test) invokes this
    module from elsewhere.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def resolve(current_path: Path, resolved_relative_path: str, root: Path) -> int:
    """Regenerate the whole inventory and copy the requested path into %A.

    Returns a process exit code: 0 for a clean resolve, non-zero to fall
    back to a normal conflicted merge for this path.
    """
    regenerated = subprocess.run(
        [sys.executable, "-m", _REGEN_MODULE],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if regenerated.returncode != 0:
        sys.stderr.write(
            "module-inventory-merge-driver: regeneration failed, falling "
            "back to a conflicted merge\n"
        )
        sys.stderr.write(regenerated.stdout)
        sys.stderr.write(regenerated.stderr)
        return regenerated.returncode

    resolved_path = root / resolved_relative_path
    if not resolved_path.is_file():
        # The merged tree no longer needs this exact shard (e.g. the entry
        # count shrank and this shard index was retired by the regeneration
        # step's own cleanup). Deleting a path is outside what a merge
        # driver can safely signal, so fall back to a normal conflicted
        # merge for this one path and let a human/agent resolve it (which,
        # per the same regeneration step, means deleting the file).
        sys.stderr.write(
            "module-inventory-merge-driver: regeneration no longer "
            f"produces {resolved_relative_path}; this shard should be "
            "deleted, which a merge driver cannot do -- falling back to a "
            "conflicted merge\n"
        )
        return 1

    current_path.write_bytes(resolved_path.read_bytes())
    return 0


def main(argv: list[str]) -> int:
    """Entry point matching git's ``driver %O %A %B %L %P`` contract."""
    if len(argv) < 5:
        sys.stderr.write("usage: module_inventory_merge_driver.py %O %A %B %L %P\n")
        return 2
    _ancestor, current, _other, _marker_size, resolved_relative_path = argv[:5]
    root = _repo_root()
    return resolve(Path(current), resolved_relative_path, root)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
