import importlib
import os
from types import ModuleType
from typing import cast

import pytest

TOOLS_CORE_REQUIRED_ENV = "TOOLS_CORE_REQUIRED"
TOOLS_CORE_MISSING_REASON = (
    "tools_core wheel not installed (run: maturin develop --features python)"
)


def import_required_tools_core() -> ModuleType:
    """Import tools_core, hard-failing only in the Rust-enabled CI lane."""
    if os.environ.get(TOOLS_CORE_REQUIRED_ENV) == "1":
        return importlib.import_module("tools_core")

    return cast(
        ModuleType,
        pytest.importorskip(
            "tools_core",
            reason=TOOLS_CORE_MISSING_REASON,
        ),
    )
