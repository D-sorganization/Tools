"""Skip lower_body_model tests when MuJoCo cannot initialize.

The ``mujoco`` Python package ships bundled plugin DLLs that fail to load
on some Windows configurations (``OSError: [WinError 1114] A dynamic link
library (DLL) initialization routine failed``). When that happens the
entire test module fails to collect, blocking pre-push hooks for every
contributor regardless of whether their change touches this module.

This conftest converts that environment-specific collection failure into a
clean skip via ``collect_ignore``. Linux CI (and any working Windows
install) imports ``mujoco`` successfully and runs the tests normally; only
broken installs skip.
"""

from __future__ import annotations

import os

collect_ignore: list[str] = []

try:  # pragma: no cover - environment dependent
    import mujoco  # noqa: F401
except (ImportError, OSError):  # pragma: no cover - environment dependent
    # Ignore every test module in this directory that imports mujoco.
    _here = os.path.dirname(__file__)
    for _name in os.listdir(_here):
        if _name.startswith("test_") and _name.endswith(".py"):
            collect_ignore.append(_name)
