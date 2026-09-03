#!/usr/bin/env python3
"""Generate every tool catalog from the one registry: ``src/**/gui_registration.py``.

The ``GUI_INFO`` dicts are the single source of truth (Tools #4916). From them
this script produces, deterministically:

1. ``tools.json`` -- Unified Launcher manifest (categorized, surface-expanded
   entries; each entry also carries ``tool_id``, ``surface`` and ``maturity``)
2. ``tool_surface_contract.json`` -- cross-repo parity contract (one entry per
   logical tool; key set is frozen for downstream consumers)
3. the ``README.md`` tool catalog table between the ``tool-catalog`` markers

Registry fields (per ``GUI_INFO``)::

    name, tool_name, description, category           required
    pyqt6: {module, class, ...}                        PyQt6 surface (needs launch_pyqt6.py)
    web: {path, port, auto_open_browser} | False       web surface (needs launch_web.py)
                                                       or an explicit "no web app"
    tkinter: {...}                                     legacy GUI (needs launch_gui.py)
    maturity: "stable" | "beta" | "experimental"      default "stable"
    help: "<repo-relative path or URL>"                default: the tool README if present
    catalog_visible: False                             keep metadata, stay out of catalogs

``--check`` fails when any generated output is stale or when a ``package.json``
web app under ``src/`` is reachable from no launcher (no ``launch_web.py`` in
an ancestor tool directory and no registration declaring ``"web": False``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CONTRACT_VERSION = "1.0.0"
MATURITIES = ("stable", "beta", "experimental")
README_START = "<!-- tool-catalog:start -->"
README_END = "<!-- tool-catalog:end -->"
_SKIP_DIRS = {"node_modules", "dist", "build", ".git"}

# Cells are separated by pipes that are not backslash-escaped: a description
# containing a literal "|" is emitted as "\|" and must not split its row.
_TABLE_CELL_SPLIT = re.compile(r"(?<!\\)\|")
# A GFM delimiter-row cell: dashes with optional alignment colons.
_SEPARATOR_CELL = re.compile(r"^:?-+:?$")
# Prettier never renders a delimiter cell narrower than three dashes.
_MIN_COLUMN_WIDTH = 3


def _emit_stdout(message: str) -> None:
    """Write a single line to stdout for the CLI contract."""
    sys.stdout.write(f"{message}\n")


@dataclass(frozen=True)
class ToolRegistration:
    """Internal representation of a discovered tool registration.

    Invariants (Design by Contract):
    - id is always a non-empty snake_case string
    - name is always a non-empty string
    - category is always a non-empty string
    - At least one surface (has_pyqt6, has_web or has_legacy_gui) is True
    - maturity is one of MATURITIES
    """

    id: str
    name: str
    description: str
    category: str
    has_pyqt6: bool
    has_web: bool
    has_legacy_gui: bool = False  # Tkinter / non-PyQt6 GUI (launch_gui.py)
    maturity: str = "stable"
    help: str | None = None
    tool_dir: str = ""


def load_gui_info(path: Path) -> dict[str, Any] | None:
    """Load GUI_INFO from a python file.

    Pre-condition: path is a valid file path.
    Post-condition: Returns the GUI_INFO dict or None if not loadable.
    """
    try:
        spec = importlib.util.spec_from_file_location(
            f"gui_reg_{path.parent.name}", path
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, "GUI_INFO", None)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load %s: %s", path, exc)
        return None


def _is_catalog_visible(info: dict[str, Any]) -> bool:
    """Return whether a registration participates in generated catalogs."""
    return info.get("catalog_visible", True) is not False


def _maturity(info: dict[str, Any], reg_file: Path) -> str:
    value = info.get("maturity", "stable")
    if value not in MATURITIES:
        raise ValueError(
            f"{reg_file}: maturity must be one of {MATURITIES}, got {value!r}"
        )
    return str(value)


def _help(info: dict[str, Any], tool_dir: Path, repo_root: Path) -> str | None:
    value = info.get("help")
    if isinstance(value, str) and value:
        return value
    readme = tool_dir / "README.md"
    if readme.is_file():
        return readme.relative_to(repo_root).as_posix()
    return None


def _registration_files(repo_root: Path) -> list[Path]:
    src_dir = repo_root / "src"
    return sorted(
        path
        for path in src_dir.glob("**/gui_registration.py")
        if not _SKIP_DIRS & set(path.parts)
    )


def _discover_registrations(repo_root: Path) -> list[ToolRegistration]:
    """Discover all gui_registration.py files and extract ToolRegistration entries.

    Pre-condition: repo_root / 'src' exists.
    Post-condition: Returns a sorted (by id) list of unique ToolRegistrations.
    """
    seen_ids: dict[str, ToolRegistration] = {}

    for reg_file in _registration_files(repo_root):
        info = load_gui_info(reg_file)
        if not info:
            continue
        if not _is_catalog_visible(info):
            continue

        tool_dir = reg_file.parent

        # Determine tool ID: prefer explicit 'tool_name', fall back to directory name
        tool_id = info.get("tool_name") or tool_dir.name

        # Determine surface availability
        has_pyqt6 = "pyqt6" in info and (tool_dir / "launch_pyqt6.py").exists()
        has_web = (
            isinstance(info.get("web"), dict) and (tool_dir / "launch_web.py").exists()
        )
        # Legacy/Tkinter GUI: explicit launch_gui.py with no PyQt6 surface
        has_legacy_gui = (
            not has_pyqt6
            and "tkinter" in info
            and (tool_dir / "launch_gui.py").exists()
        )

        # Skip tools with no available surface
        if not has_pyqt6 and not has_web and not has_legacy_gui:
            continue

        registration = ToolRegistration(
            id=tool_id,
            name=info.get("name", "Unknown Tool"),
            description=info.get("description", ""),
            category=info.get("category", "Uncategorized"),
            has_pyqt6=has_pyqt6,
            has_web=has_web,
            has_legacy_gui=has_legacy_gui,
            maturity=_maturity(info, reg_file),
            help=_help(info, tool_dir, repo_root),
            tool_dir=tool_dir.relative_to(repo_root).as_posix(),
        )

        if tool_id in seen_ids:
            logger.warning(
                "Duplicate tool_name '%s' found, skipping %s", tool_id, reg_file
            )
        else:
            seen_ids[tool_id] = registration

    return sorted(seen_ids.values(), key=lambda r: r.id)


def _manifest_entry(
    reg: ToolRegistration, name: str, surface: str, script: str
) -> dict[str, Any]:
    return {
        "name": name,
        "path": f"{reg.tool_dir}/{script}",
        "type": "python",
        "desc": reg.description,
        "tool_id": reg.id,
        "surface": surface,
        "maturity": reg.maturity,
    }


def generate_manifest_data(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    """Generate manifest data (tools.json format) from gui_registration.py files.

    Pre-condition: repo_root is a valid repo root with src/ directory.
    Post-condition: Returns a dict keyed by category with sorted tool entries.
    """
    registrations = _discover_registrations(repo_root)
    manifest: dict[str, list[dict[str, Any]]] = {}

    for reg in registrations:
        entries = manifest.setdefault(reg.category, [])
        if reg.has_pyqt6:
            name = f"{reg.name} (PyQt6)" if reg.has_web else reg.name
            entries.append(_manifest_entry(reg, name, "pyqt6", "launch_pyqt6.py"))
        if reg.has_web:
            entries.append(
                _manifest_entry(reg, f"{reg.name} (Web)", "web", "launch_web.py")
            )
        if reg.has_legacy_gui:
            entries.append(
                _manifest_entry(reg, reg.name, "legacy_gui", "launch_gui.py")
            )

    # Sort categories and tools for determinism
    sorted_manifest: dict[str, list[dict[str, Any]]] = {}
    for cat in sorted(manifest.keys()):
        tools = manifest[cat]
        tools.sort(key=lambda x: x["name"])
        sorted_manifest[cat] = tools

    return sorted_manifest


def _find_tool_dir(src_dir: Path, tool_id: str) -> Path | None:
    """Find the directory containing a tool's gui_registration.py by its ID."""
    for reg_file in _registration_files(src_dir.parent):
        info = load_gui_info(reg_file)
        if not info:
            continue
        if not _is_catalog_visible(info):
            continue
        candidate_id = info.get("tool_name") or reg_file.parent.name
        if candidate_id == tool_id:
            return reg_file.parent
    return None


def generate_contract_data(repo_root: Path) -> dict[str, Any]:
    """Generate tool surface contract data for cross-repo parity checking.

    Pre-condition: repo_root is a valid repo root with src/ directory.
    Post-condition: Returns a dict conforming to tool_surface_contract.schema.json:
      - "version": semver string
      - "tools": list of tool entries sorted by id, each with:
        - "id": snake_case tool identifier
        - "name": human-readable display name
        - "description": tool description
        - "category": tool category
        - "surfaces": {"pyqt6": bool, "web": bool, "legacy_gui": bool}
    The key set is frozen: downstream repos compare it verbatim.
    """
    registrations = _discover_registrations(repo_root)

    tools: list[dict[str, Any]] = []
    for reg in registrations:
        tools.append(
            {
                "id": reg.id,
                "name": reg.name,
                "description": reg.description,
                "category": reg.category,
                "surfaces": {
                    "pyqt6": reg.has_pyqt6,
                    "web": reg.has_web,
                    "legacy_gui": reg.has_legacy_gui,
                },
            }
        )

    return {
        "version": CONTRACT_VERSION,
        "tools": tools,
    }


def _surfaces_label(reg: ToolRegistration) -> str:
    labels = []
    if reg.has_pyqt6:
        labels.append("PyQt6")
    if reg.has_web:
        labels.append("Web")
    if reg.has_legacy_gui:
        labels.append("Tk")
    return " + ".join(labels)


def _display_width(text: str) -> int:
    """Rendered column width of ``text``, counting East Asian wide cells as two.

    This is the same width rule Prettier's markdown table printer uses, so a
    table padded with it survives the pre-commit formatter untouched.
    """
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def _split_table_row(line: str) -> list[str]:
    """Split one pipe table row into stripped cells.

    Pre-condition: ``line`` is a pipe table row delimited by leading and
    trailing pipes. Raises ``ValueError`` otherwise.
    """
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        raise ValueError(f"not a pipe table row: {line!r}")
    return [cell.strip() for cell in _TABLE_CELL_SPLIT.split(stripped[1:-1])]


def _align_markdown_table(table: str) -> str:
    """Pad a pipe table's cells to Prettier's markdown table layout.

    Every column is widened to its widest cell (minimum three characters) and
    the delimiter row is filled with dashes to the same width, which is exactly
    what the repo's ``prettier`` pre-commit hook produces. Generating the table
    in this shape keeps the committed README byte-identical to the formatter's
    output, so the freshness gate is not fighting the formatter.

    Post-condition: the result is idempotent under a second call.
    """
    rows = [_split_table_row(line) for line in table.strip("\n").split("\n")]
    if len(rows) < 2:
        raise ValueError("markdown table needs a header and a delimiter row")
    columns = len(rows[0])
    if any(len(row) != columns for row in rows):
        raise ValueError("markdown table rows disagree on column count")
    widths = [
        max(
            _MIN_COLUMN_WIDTH,
            *(
                _display_width(row[index])
                for position, row in enumerate(rows)
                if position != 1
            ),
        )
        for index in range(columns)
    ]
    lines = []
    for position, row in enumerate(rows):
        if position == 1:
            cells = ["-" * widths[index] for index in range(columns)]
        else:
            cells = [
                row[index] + " " * (widths[index] - _display_width(row[index]))
                for index in range(columns)
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _normalise_markdown_table(table: str) -> tuple[tuple[str, ...], ...]:
    """Reduce a pipe table to its content, ignoring padding only.

    Cell padding, internal runs of whitespace and delimiter-row dash counts are
    normalised away; cell text, column count, column order and row order are
    all preserved, so two tables compare equal exactly when they say the same
    thing in the same order.

    Raises ``ValueError`` when ``table`` is not a well-formed pipe table.
    """
    lines = [line for line in table.strip("\n").split("\n") if line.strip()]
    if len(lines) < 2:
        raise ValueError("markdown table needs a header and a delimiter row")
    rows = [
        tuple(" ".join(cell.split()) for cell in _split_table_row(line))
        for line in lines
    ]
    columns = len(rows[0])
    if any(len(row) != columns for row in rows):
        raise ValueError("markdown table rows disagree on column count")
    separator = rows[1]
    if not all(_SEPARATOR_CELL.match(cell) for cell in separator):
        raise ValueError(f"markdown table delimiter row is malformed: {separator!r}")
    canonical_separator = tuple(
        f"{':' if cell.startswith(':') else ''}-{':' if cell.endswith(':') else ''}"
        for cell in separator
    )
    return (rows[0], canonical_separator, *rows[2:])


def generate_readme_catalog(repo_root: Path) -> str:
    """Render the README tool catalog table (one row per launcher-registered tool).

    Post-condition: the table is padded to Prettier's layout, so writing it and
    then running the pre-commit markdown formatter is a no-op.
    """
    registrations = _discover_registrations(repo_root)
    lines = [
        "| Tool | Category | Surfaces | Maturity | What it does | Help |",
        "| ---- | -------- | -------- | -------- | ------------ | ---- |",
    ]
    for reg in sorted(registrations, key=lambda r: (r.category, r.id)):
        help_cell = f"[docs]({reg.help})" if reg.help else "—"
        description = reg.description.replace("|", "\\|")
        lines.append(
            f"| `{reg.id}` | {reg.category} | {_surfaces_label(reg)} | "
            f"{reg.maturity} | {description} | {help_cell} |"
        )
    return _align_markdown_table("\n".join(lines) + "\n")


def _readme_split(text: str) -> tuple[str, str, str]:
    start = text.find(README_START)
    end = text.find(README_END)
    if start < 0 or end < 0 or end < start:
        raise ValueError(
            f"README.md is missing the {README_START}/{README_END} markers"
        )
    head = text[: start + len(README_START)] + "\n"
    body = text[start + len(README_START) + 1 : end]
    return head, body, text[end:]


def readme_catalog_is_fresh(repo_root: Path) -> bool:
    """Whether the committed README catalog table says what the registry says.

    The comparison is on normalised table structure, not on bytes: the
    pre-commit markdown formatter owns cell padding and the blank lines around
    the table, so comparing raw strings would make the gate unpassable
    regardless of content. Content, column count and row order are still
    compared exactly, so a changed, added, removed or reordered tool is stale.
    """
    readme = repo_root / "README.md"
    if not readme.is_file():
        return False
    _head, body, _tail = _readme_split(
        readme.read_text(encoding="utf-8").replace("\r\n", "\n")
    )
    try:
        committed = _normalise_markdown_table(body)
    except ValueError:
        return False
    return committed == _normalise_markdown_table(generate_readme_catalog(repo_root))


def write_readme_catalog(repo_root: Path) -> None:
    """Rewrite the README catalog table in the formatter's own layout.

    The table is surrounded by blank lines and padded to Prettier's widths so
    the pre-commit markdown hook has nothing left to change.
    """
    readme = repo_root / "README.md"
    head, _body, tail = _readme_split(
        readme.read_text(encoding="utf-8").replace("\r\n", "\n")
    )
    readme.write_text(
        head + "\n" + generate_readme_catalog(repo_root) + "\n" + tail,
        encoding="utf-8",
        newline="\n",
    )


def unreachable_web_apps(repo_root: Path) -> list[str]:
    """Return ``package.json`` web apps under src/ that no launcher can reach.

    A package.json is reachable when an ancestor directory (up to src/) holds a
    ``launch_web.py`` or a ``gui_registration.py`` whose ``GUI_INFO["web"]`` is
    ``False`` (an explicit "not a launcher tile" decision).
    """
    src_dir = repo_root / "src"
    unreachable: list[str] = []
    for package_json in sorted(src_dir.glob("**/package.json")):
        if _SKIP_DIRS & set(package_json.parts):
            continue
        reachable = False
        for ancestor in package_json.parents:
            if ancestor == src_dir.parent:
                break
            if (ancestor / "launch_web.py").is_file():
                reachable = True
                break
            registration = ancestor / "gui_registration.py"
            if registration.is_file():
                info = load_gui_info(registration) or {}
                if info.get("web") is False:
                    reachable = True
                    break
            if ancestor == src_dir:
                break
        if not reachable:
            unreachable.append(package_json.relative_to(repo_root).as_posix())
    return unreachable


def _serialize_manifest(manifest: dict[str, list[dict[str, Any]]]) -> str:
    return json.dumps(manifest, indent=4) + "\n"


def _serialize_contract(contract: dict[str, Any]) -> str:
    return json.dumps(contract, indent=2) + "\n"


def check(repo_root: Path) -> list[str]:
    """Return a list of stale-output diagnostics (empty when everything is fresh)."""
    problems: list[str] = []
    manifest_path = repo_root / "tools.json"
    contract_path = repo_root / "tool_surface_contract.json"
    expected_manifest = _serialize_manifest(generate_manifest_data(repo_root))
    expected_contract = _serialize_contract(generate_contract_data(repo_root))
    for path, expected in (
        (manifest_path, expected_manifest),
        (contract_path, expected_contract),
    ):
        actual = (
            path.read_text(encoding="utf-8").replace("\r\n", "\n")
            if path.is_file()
            else ""
        )
        if actual != expected:
            problems.append(f"{path.name} is stale")
    try:
        if not readme_catalog_is_fresh(repo_root):
            problems.append("README.md tool catalog table is stale")
    except ValueError as exc:
        problems.append(str(exc))
    for package_json in unreachable_web_apps(repo_root):
        problems.append(
            f"{package_json}: web app reachable from no launcher (add launch_web.py "
            'or declare "web": False in gui_registration.py)'
        )
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    """Generate tools.json, tool_surface_contract.json and the README catalog."""
    parser = argparse.ArgumentParser(description="Tool registry generator")
    parser.add_argument(
        "--check", action="store_true", help="fail if any generated output is stale"
    )
    parser.add_argument("--root", type=Path, default=None, help=argparse.SUPPRESS)
    if argv is None:
        # Only read the process arguments when run as this script; an imported
        # ``main()`` (tests, governance wrappers) gets the default behaviour.
        own_script = Path(sys.argv[0]).name == Path(__file__).name
        argv = sys.argv[1:] if own_script else []
    args = parser.parse_args(argv)
    repo_root = (
        args.root.resolve()
        if args.root is not None
        else Path(__file__).resolve().parents[1]
    )

    if args.check:
        problems = check(repo_root)
        for problem in problems:
            sys.stderr.write(f"ERROR: {problem}\n")
        if problems:
            sys.stderr.write("Run: python scripts/generate_tools_json.py\n")
            return 1
        _emit_stdout("tool registry outputs are fresh")
        return 0

    # Generate manifest
    manifest = generate_manifest_data(repo_root)
    tools_json_path = repo_root / "tools.json"
    with open(tools_json_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(_serialize_manifest(manifest))

    tool_count = sum(len(v) for v in manifest.values())
    logger.info("Generated tools.json with %d tools.", tool_count)

    # Generate contract
    contract = generate_contract_data(repo_root)
    contract_path = repo_root / "tool_surface_contract.json"
    with open(contract_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(_serialize_contract(contract))

    logger.info(
        "Generated tool_surface_contract.json with %d tools.",
        len(contract["tools"]),
    )

    if (repo_root / "README.md").is_file():
        try:
            write_readme_catalog(repo_root)
        except ValueError as exc:
            logger.warning("README catalog not written: %s", exc)

    # Keep the CLI contract explicit for CI callers without using raw print().
    _emit_stdout(f"Generated tools.json with {tool_count} tools.")
    _emit_stdout(
        f"Generated tool_surface_contract.json with {len(contract['tools'])} tools."
    )
    unreachable = unreachable_web_apps(repo_root)
    for package_json in unreachable:
        logger.warning("web app reachable from no launcher: %s", package_json)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
