#!/usr/bin/env python3
"""Register this repo's local-only git merge drivers.

``.gitattributes`` can only *name* a merge driver
(``manuals/tools/manifests/module-inventory.json merge=module-inventory-regen``);
it cannot embed the command that driver actually runs. That split is
deliberate on git's part: it is a security boundary. If an attribute alone
could make a fresh clone execute an arbitrary command, cloning an untrusted
repository and running ``git merge`` would be enough to run arbitrary code.
The command has to live in *local*, per-clone config
(``git config merge.<name>.driver``), which nothing about cloning or
checking out the repo sets up on its own.

This script performs that one-time local registration. It is idempotent
(safe to run repeatedly) and is wired into ``scripts/setup_hooks.py``, this
repo's existing "install local git-level automation" entry point, so it
runs automatically for anyone who runs the documented hook setup rather
than requiring a separate manual step nobody remembers to run. It can also
be run directly:

    python3 scripts/git/install_merge_drivers.py

See ``scripts/git/module_inventory_merge_driver.py`` for what the driver
itself does once registered.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DRIVER_NAME = "module-inventory-regen"
DRIVER_SCRIPT = Path("scripts/git/module_inventory_merge_driver.py")
DRIVER_DESCRIPTION = (
    "Regenerate the Tools module inventory from the merged tree "
    "(see scripts/git/module_inventory_merge_driver.py)"
)


def _git_config(key: str, value: str) -> None:
    subprocess.run(["git", "config", key, value], check=True)


def install(python_executable: str = sys.executable) -> str:
    """Register the module-inventory merge driver in local git config.

    Returns the exact driver command string that was registered, mainly so
    callers/tests can assert on it without re-deriving the quoting rules.
    """
    driver_command = (
        f'"{Path(python_executable).as_posix()}" '
        f"{DRIVER_SCRIPT.as_posix()} %O %A %B %L %P"
    )
    _git_config(f"merge.{DRIVER_NAME}.driver", driver_command)
    _git_config(f"merge.{DRIVER_NAME}.name", DRIVER_DESCRIPTION)
    return driver_command


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    driver_command = install()
    logger.info(
        "[OK] registered git merge driver '%s' -> %s", DRIVER_NAME, driver_command
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
