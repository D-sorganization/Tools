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


def _squash_table_padding(text: str) -> str:
    """Re-pad every table row to a single space per cell (content untouched)."""
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped[1:-1].split("|")]
            if all(set(cell) <= {"-", ":"} and cell for cell in cells):
                cells = ["-" * (index + 1) for index in range(len(cells))]
            lines.append("| " + " | ".join(cells) + " |")
        else:
            lines.append(line)
    return "\n".join(lines)


def test_readme_catalog_round_trip_and_check(gen: ModuleType, repo: Path) -> None:
    assert "README.md tool catalog table is stale" in gen.check(repo)
    gen.write_readme_catalog(repo)
    rows = gen._normalise_markdown_table(gen.generate_readme_catalog(repo))
    assert (
        "`alpha`",
        "Cat",
        "PyQt6 + Web",
        "beta",
        "A",
        "[docs](src/alpha/README.md)",
    ) in rows
    assert ("`beta`", "Cat", "PyQt6", "stable", "B", "—") in rows
    assert gen.readme_catalog_is_fresh(repo)


def test_generated_catalog_is_prettier_aligned_and_idempotent(
    gen: ModuleType, repo: Path
) -> None:
    """The generator emits the formatter's own layout, so re-padding is a no-op."""
    table = gen.generate_readme_catalog(repo)
    assert gen._align_markdown_table(table) == table
    header, delimiter = table.split("\n")[:2]
    # Aligned means every row is the same rendered width as the header.
    assert all(len(line) == len(header) for line in table.strip("\n").split("\n")), (
        table
    )
    assert set(delimiter) <= {"|", " ", "-"}


def test_reformatting_padding_keeps_the_gate_green(gen: ModuleType, repo: Path) -> None:
    """Regression (#4916): the markdown formatter owns padding, not the gate.

    The old gate string-compared the committed table against the generated one,
    so any re-padding by the pre-commit markdown hook made it permanently red.
    """
    assert gen.main(["--root", str(repo)]) == 0
    assert gen.check(repo) == []
    readme = repo / "README.md"
    original = readme.read_text(encoding="utf-8")
    readme.write_text(_squash_table_padding(original), encoding="utf-8", newline="\n")
    assert readme.read_text(encoding="utf-8") != original, "padding was not changed"
    assert gen.readme_catalog_is_fresh(repo)
    assert gen.check(repo) == []


def test_changing_a_row_content_makes_the_gate_red(gen: ModuleType, repo: Path) -> None:
    """The gate stays strict about content, column count and row order."""
    gen.write_readme_catalog(repo)
    readme = repo / "README.md"
    fresh = readme.read_text(encoding="utf-8")

    # (a) a changed cell
    readme.write_text(fresh.replace("beta ", "gamma"), encoding="utf-8", newline="\n")
    assert not gen.readme_catalog_is_fresh(repo)

    # (b) a dropped row
    lines = fresh.split("\n")
    dropped = [line for line in lines if not line.startswith("| `beta`")]
    assert len(dropped) < len(lines)
    readme.write_text("\n".join(dropped), encoding="utf-8", newline="\n")
    assert not gen.readme_catalog_is_fresh(repo)

    # (c) reordered rows
    body = [index for index, line in enumerate(lines) if line.startswith("| `")]
    reordered = list(lines)
    reordered[body[0]], reordered[body[-1]] = (
        reordered[body[-1]],
        reordered[body[0]],
    )
    readme.write_text("\n".join(reordered), encoding="utf-8", newline="\n")
    assert not gen.readme_catalog_is_fresh(repo)

    # (d) a dropped column
    narrowed = [
        (
            "| " + " | ".join(line.strip()[1:-1].split("|")[:-1]).strip() + " |"
            if line.startswith("| ")
            else line
        )
        for line in lines
    ]
    readme.write_text("\n".join(narrowed), encoding="utf-8", newline="\n")
    assert not gen.readme_catalog_is_fresh(repo)

    readme.write_text(fresh, encoding="utf-8", newline="\n")
    assert gen.readme_catalog_is_fresh(repo)


def test_escaped_pipe_in_a_description_does_not_split_its_row(
    gen: ModuleType, tmp_path: Path
) -> None:
    """A description containing "|" is escaped and must stay one cell."""
    tool = tmp_path / "src" / "piped"
    _write_registration(
        tool,
        '{"name": "Piped", "tool_name": "piped", "description": "a | b", '
        '"category": "Cat", "pyqt6": {"module": "m", "class": "C"}}',
    )
    (tool / "launch_pyqt6.py").touch()
    (tmp_path / "README.md").write_text(
        f"# R\n\n{gen.README_START}\n{gen.README_END}\n", encoding="utf-8"
    )
    gen.write_readme_catalog(tmp_path)
    rows = gen._normalise_markdown_table(gen.generate_readme_catalog(tmp_path))
    assert all(len(row) == 6 for row in rows)
    assert ("`piped`", "Cat", "PyQt6", "stable", "a \\| b", "—") in rows
    assert gen.readme_catalog_is_fresh(tmp_path)


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
