"""themes.json is the single source of theme colors (issue #5917).

A color changed in ``themes.json`` alone must reach the Python theme loader and
the design-token API, and ``design_tokens.json`` must not drift from it.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
THEMES_JSON = REPO_ROOT / "src" / "shared" / "theme-definitions" / "themes.json"
TOKENS_JSON = REPO_ROOT / "src" / "shared" / "design_tokens.json"
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_design_tokens.py"
NEW_ACCENT = "#123456"


def _load_sync_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sync_design_tokens", SYNC_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def edited_catalog(tmp_path: Path) -> Path:
    """A copy of themes.json whose light accent is changed."""
    catalog = json.loads(THEMES_JSON.read_text(encoding="utf-8"))
    catalog["themes"]["light"]["colors"]["accent"] = NEW_ACCENT
    path = tmp_path / "themes.json"
    path.write_text(json.dumps(catalog), encoding="utf-8")
    return path


def test_design_tokens_json_is_in_sync() -> None:
    """CI drift check: the committed design_tokens.json matches themes.json."""
    sync = _load_sync_script()
    assert sync.main(["--check"]) == 0


def test_sync_check_fails_on_drift(edited_catalog: Path, tmp_path: Path) -> None:
    sync = _load_sync_script()
    tokens = tmp_path / "design_tokens.json"
    tokens.write_text(TOKENS_JSON.read_text(encoding="utf-8"), encoding="utf-8")
    args = ["--themes", str(edited_catalog), "--tokens", str(tokens)]

    assert sync.main([*args, "--check"]) == 1
    assert sync.main(args) == 0
    regenerated = json.loads(tokens.read_text(encoding="utf-8"))
    assert regenerated["themes"]["light"]["accent"] == NEW_ACCENT
    assert sync.main([*args, "--check"]) == 0


def test_sync_preserves_non_theme_tokens(edited_catalog: Path) -> None:
    sync = _load_sync_script()
    tokens = json.loads(TOKENS_JSON.read_text(encoding="utf-8"))
    catalog = json.loads(edited_catalog.read_text(encoding="utf-8"))

    rebuilt = sync.build_design_tokens(catalog, copy.deepcopy(tokens))

    assert list(rebuilt) == list(tokens)
    for section in ("spacing", "typography", "radii"):
        assert rebuilt[section] == tokens[section]


def test_design_token_api_reads_catalog(
    edited_catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shared.python.sidekick.ui import design_tokens

    monkeypatch.setattr(design_tokens, "_THEMES_JSON", edited_catalog)
    assert design_tokens.get_token_dict("light")["@color_accent"] == NEW_ACCENT
    assert f"qproperty-accent: {NEW_ACCENT};" in design_tokens.get_qss_variables()


def test_python_theme_loader_reads_catalog(
    edited_catalog: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shared.python.theme import colors

    monkeypatch.setattr(colors, "_THEMES_JSON", edited_catalog)
    themes = colors._load_themes_from_json()
    assert themes is not None
    assert themes["Light"]["accent"] == NEW_ACCENT


def test_theme_resolution_prefers_vendor_in_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When theme module is placed in a consumer repository, resolve vendor/ud-tools."""
    from shared.python.theme import colors

    consumer_root = tmp_path / "ConsumerRepo"
    module_dir = consumer_root / "src" / "shared" / "python" / "theme"
    module_dir.mkdir(parents=True)
    fake_module_file = module_dir / "colors.py"
    fake_module_file.touch()

    # Local stale catalog
    stale_catalog = (
        consumer_root / "src" / "shared" / "theme-definitions" / "themes.json"
    )
    stale_catalog.parent.mkdir(parents=True)
    stale_catalog.write_text("STALE", encoding="utf-8")

    # Vendored canonical catalog
    vendor_catalog = (
        consumer_root
        / "vendor"
        / "ud-tools"
        / "src"
        / "shared"
        / "theme-definitions"
        / "themes.json"
    )
    vendor_catalog.parent.mkdir(parents=True)
    vendor_catalog.write_text("CANONICAL", encoding="utf-8")

    monkeypatch.setattr(colors, "__file__", str(fake_module_file))
    resolved = colors._find_canonical_themes_json()
    assert resolved == vendor_catalog
    assert resolved.read_text(encoding="utf-8") == "CANONICAL"


def test_theme_package_exports_declared() -> None:
    """Root package.json and theme package.json expose canonical exports."""
    root_pkg = json.loads((REPO_ROOT / "package.json").read_text(encoding="utf-8"))
    assert "src/shared/typescript/theme" in root_pkg.get("workspaces", [])
    expected_theme_export = "./src/shared/typescript/theme/index.ts"
    assert root_pkg.get("exports", {}).get("./theme") == expected_theme_export

    theme_pkg_file = (
        REPO_ROOT / "src" / "shared" / "typescript" / "theme" / "package.json"
    )
    assert theme_pkg_file.is_file()
    theme_pkg = json.loads(theme_pkg_file.read_text(encoding="utf-8"))
    assert theme_pkg.get("name") == "@d-sorganization/ud-tools-theme"
    assert theme_pkg.get("main") == "index.ts"
