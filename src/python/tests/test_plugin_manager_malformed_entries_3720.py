"""Regression tests for #3720.

``PluginManager.load_tools`` previously caught only ``KeyError`` in its
per-item loop. A non-dict entry (a bare string or list) makes
``_build_validated_tool`` raise ``TypeError`` on ``item["path"]``, which
escaped the inner handler and bubbled to the outer
``except (KeyError, ValueError, TypeError)`` — aborting the whole loop and
returning ``{}``. A single malformed ``tools.json`` entry thus wiped the
entire tool registry.

These tests prove a malformed entry is now skipped-and-logged while valid
tools in the same and other categories survive.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def _import_plugin_manager_module() -> ModuleType | None:
    """Import plugin_manager in isolation, resolving its relative imports.

    Returns ``None`` (so callers can skip) if the module cannot be loaded.
    """
    pm_path = Path(__file__).parent.parent / "src" / "core" / "plugin_manager.py"
    if not pm_path.exists():
        return None

    mock_utils = MagicMock()
    mock_utils.safe_read_json = MagicMock(return_value={})
    mock_core_utils = MagicMock()
    mock_core_utils.file_utils = mock_utils

    fake_pkg_name = "plugin_manager_iss3720_pkg"
    fake_core_name = f"{fake_pkg_name}.core"
    fake_utils_name = f"{fake_pkg_name}.utils"
    fake_file_utils_name = f"{fake_pkg_name}.utils.file_utils"

    fake_pkg = MagicMock()
    fake_pkg.__name__ = fake_pkg_name
    fake_pkg.__path__ = [str(pm_path.parent.parent)]
    fake_pkg.__package__ = fake_pkg_name

    fake_core = MagicMock()
    fake_core.__name__ = fake_core_name
    fake_core.__path__ = [str(pm_path.parent)]
    fake_core.__package__ = fake_pkg_name

    fake_file_utils = MagicMock()
    fake_file_utils.safe_read_json = MagicMock(return_value={})

    extra_modules = {
        fake_pkg_name: fake_pkg,
        fake_core_name: fake_core,
        fake_utils_name: mock_core_utils,
        fake_file_utils_name: fake_file_utils,
    }

    try:
        spec = importlib.util.spec_from_file_location(
            f"{fake_core_name}.plugin_manager",
            pm_path,
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        module.__package__ = fake_core_name
        with patch.dict("sys.modules", extra_modules):
            spec.loader.exec_module(module)
        return module
    except Exception:  # noqa: BLE001 — test isolation: skip if import fails
        return None


def _make_pm_with_config(tmp_path: Path, config: dict[str, Any]) -> Any:
    """Build a PluginManager whose load_tools reads ``config`` directly."""
    module = _import_plugin_manager_module()
    if module is None:
        return None

    (tmp_path / "tools").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tools" / "good.py").write_text("# tool\n", encoding="utf-8")
    (tmp_path / "tools" / "other.py").write_text("# tool\n", encoding="utf-8")

    pm = module.PluginManager(repo_root=tmp_path)
    pm.tools_file = tmp_path / "tools.json"
    pm.tools_file.write_text(json.dumps(config), encoding="utf-8")

    def _read(path: Any, default: Any = None) -> Any:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    # Rebind the module-global safe_read_json to a real reader for this call.
    module.safe_read_json = _read  # type: ignore[attr-defined]
    return pm


def test_load_tools_non_dict_entry_does_not_wipe_registry(tmp_path: Path) -> None:
    """A bare-string entry is skipped; valid tools survive (#3720)."""
    config: dict[str, Any] = {
        "Cat": [
            "bare-string",
            {
                "name": "Good",
                "path": "tools/good.py",
                "type": "python",
                "desc": "ok",
            },
        ],
        "Other": [
            {
                "name": "Other",
                "path": "tools/other.py",
                "type": "python",
                "desc": "ok",
            }
        ],
    }
    pm = _make_pm_with_config(tmp_path, config)
    if pm is None:
        pytest.skip("PluginManager not importable in isolation")

    result = pm.load_tools()

    assert result != {}, "malformed entry must not wipe the whole registry"
    assert [t.name for t in result["Cat"]] == ["Good"]
    assert [t.name for t in result["Other"]] == ["Other"]


def test_load_tools_list_entry_is_skipped_and_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A list (non-dict) entry is skipped-and-logged while siblings survive."""
    config: dict[str, Any] = {
        "Cat": [
            ["a.py"],
            {
                "name": "Good",
                "path": "tools/good.py",
                "type": "python",
                "desc": "ok",
            },
        ]
    }
    pm = _make_pm_with_config(tmp_path, config)
    if pm is None:
        pytest.skip("PluginManager not importable in isolation")

    with caplog.at_level(logging.WARNING):
        result = pm.load_tools()

    assert [t.name for t in result["Cat"]] == ["Good"]
    assert any(
        "non-dict tool entry" in rec.message and "Cat" in rec.message
        for rec in caplog.records
    )
