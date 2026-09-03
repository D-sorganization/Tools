"""One registry, one launcher (Tools #4916): generated outputs stay in step."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate_tools_json_registry", REPO_ROOT / "scripts" / "generate_tools_json.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_tools_json_registry"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gen() -> ModuleType:
    return _load()


def _write_registration(tool_dir: Path, body: str) -> None:
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / "gui_registration.py").write_text(
        f"GUI_INFO = {body}\ndef get_gui_info():\n    return GUI_INFO\n",
        encoding="utf-8",
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    alpha = src / "alpha"
    _write_registration(
        alpha,
        '{"name": "Alpha", "tool_name": "alpha", "description": "A", '
        '"category": "Cat", "maturity": "beta", "pyqt6": {"module": "m", '
        '"class": "C"}, "web": {"port": 1}}',
    )
    (alpha / "launch_pyqt6.py").touch()
    (alpha / "launch_web.py").touch()
    (alpha / "web").mkdir()
    (alpha / "web" / "package.json").write_text("{}", encoding="utf-8")
    (alpha / "README.md").write_text("# Alpha\n", encoding="utf-8")
    beta = src / "beta"
    _write_registration(
        beta,
        '{"name": "Beta", "tool_name": "beta", "description": "B", '
        '"category": "Cat", "pyqt6": {"module": "m", "class": "C"}, "web": False}',
    )
    (beta / "launch_pyqt6.py").touch()
    (beta / "frontend").mkdir()
    (beta / "frontend" / "package.json").write_text("{}", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        f"# R\n\n{_load().README_START}\n{_load().README_END}\n", encoding="utf-8"
    )
    return tmp_path


def test_repository_outputs_are_fresh(gen: ModuleType) -> None:
    assert gen.check(REPO_ROOT) == []


def test_repository_registry_agrees_across_all_three_outputs(gen: ModuleType) -> None:
    manifest = json.loads((REPO_ROOT / "tools.json").read_text(encoding="utf-8"))
    contract = json.loads(
        (REPO_ROOT / "tool_surface_contract.json").read_text(encoding="utf-8")
    )
    manifest_ids = {entry["tool_id"] for tools in manifest.values() for entry in tools}
    contract_ids = {tool["id"] for tool in contract["tools"]}
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    readme_ids = {
        line.split("`")[1]
        for line in readme.splitlines()
        if line.startswith("| `") and "| " in line
    }
    assert manifest_ids == contract_ids
    assert contract_ids <= readme_ids
    assert "data_explorer" in contract_ids
    for tools in manifest.values():
        assert all(entry["maturity"] in gen.MATURITIES for entry in tools)


def test_every_package_json_web_app_is_reachable(gen: ModuleType) -> None:
    assert gen.unreachable_web_apps(REPO_ROOT) == []


def test_legacy_tile_launcher_is_gone() -> None:
    assert not (REPO_ROOT / "run_tile_launcher.py").exists()
    assert not (REPO_ROOT / "src" / "python" / "src" / "tile_launcher").exists()


def test_manifest_entries_carry_registry_fields(gen: ModuleType, repo: Path) -> None:
    manifest = gen.generate_manifest_data(repo)
    entries = manifest["Cat"]
    by_name = {entry["name"]: entry for entry in entries}
    assert set(by_name) == {"Alpha (PyQt6)", "Alpha (Web)", "Beta"}
    assert by_name["Alpha (Web)"]["surface"] == "web"
    assert by_name["Alpha (Web)"]["maturity"] == "beta"
    assert by_name["Beta"]["maturity"] == "stable"
    assert by_name["Beta"]["tool_id"] == "beta"


def test_contract_key_set_is_frozen(gen: ModuleType, repo: Path) -> None:
    contract = gen.generate_contract_data(repo)
    for tool in contract["tools"]:
        assert set(tool) == {"id", "name", "description", "category", "surfaces"}
        assert set(tool["surfaces"]) == {"pyqt6", "web", "legacy_gui"}


def test_readme_catalog_round_trip_and_check(gen: ModuleType, repo: Path) -> None:
    assert "README.md tool catalog table is stale" in gen.check(repo)
    gen.write_readme_catalog(repo)
    table = gen.generate_readme_catalog(repo)
    alpha_row = (
        "| `alpha` | Cat | PyQt6 + Web | beta | A | [docs](src/alpha/README.md) |"
    )
    assert alpha_row in table
    assert "| `beta` | Cat | PyQt6 | stable | B | — |" in table
    assert gen.readme_catalog_is_fresh(repo)


def test_web_false_marks_package_json_reachable(gen: ModuleType, repo: Path) -> None:
    assert gen.unreachable_web_apps(repo) == []
    orphan = repo / "src" / "gamma" / "web"
    orphan.mkdir(parents=True)
    (orphan / "package.json").write_text("{}", encoding="utf-8")
    assert gen.unreachable_web_apps(repo) == ["src/gamma/web/package.json"]
    problems = gen.check(repo)
    assert any("src/gamma/web/package.json" in problem for problem in problems)


def test_invalid_maturity_is_rejected(gen: ModuleType, tmp_path: Path) -> None:
    tool = tmp_path / "src" / "bad"
    _write_registration(
        tool,
        '{"name": "Bad", "tool_name": "bad", "description": "x", "category": "C", '
        '"maturity": "shiny", "pyqt6": {"module": "m", "class": "C"}}',
    )
    (tool / "launch_pyqt6.py").touch()
    with pytest.raises(ValueError, match="maturity"):
        gen.generate_contract_data(tmp_path)


def test_check_cli_reports_fresh(gen: ModuleType, repo: Path) -> None:
    assert gen.main(["--root", str(repo)]) == 0
    assert gen.main(["--root", str(repo), "--check"]) == 0
    (repo / "tools.json").write_text("{}\n", encoding="utf-8")
    assert gen.main(["--root", str(repo), "--check"]) == 1
