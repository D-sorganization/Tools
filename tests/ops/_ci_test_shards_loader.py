"""Load ``scripts/ci_test_shards.py`` without requiring the scripts package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SHARDS_SCRIPT = _REPO_ROOT / "scripts" / "ci_test_shards.py"


def load_ci_test_shards_module() -> Any:
    scripts_dir = str(_SHARDS_SCRIPT.parent)
    inserted = scripts_dir not in sys.path
    if inserted:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("ci_test_shards", _SHARDS_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if inserted:
        sys.path.remove(scripts_dir)
    return module
