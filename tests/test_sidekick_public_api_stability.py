"""AST-based public API stability tests for the Sidekick package (issues #3032).

Parses the public sidekick modules using the Python ast module to verify
that the public API surface area (defined via __all__) and signatures have
not changed.
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any

import pytest

log = logging.getLogger(__name__)

import os

REPO_ROOT = Path(__file__).resolve().parents[1]
SIDEKICK_ROOT = REPO_ROOT / "src" / "shared" / "python" / "sidekick"
BASELINE_PATH = REPO_ROOT / "tests" / "sidekick_api_baseline.json"


def _find_all_sidekick_modules() -> list[str]:
    test_files = []
    for root, _dirs, files in os.walk(SIDEKICK_ROOT):
        # Skip tests directories under sidekick
        parts = Path(root).relative_to(SIDEKICK_ROOT).parts
        if "tests" in parts or "__pycache__" in parts:
            continue
        for f in files:
            if f.endswith(".py"):
                full_path = Path(root) / f
                rel_path = full_path.relative_to(SIDEKICK_ROOT)
                test_files.append(rel_path.as_posix())
    return sorted(test_files)


TEST_FILES = _find_all_sidekick_modules()


def resolve_module_file(module_name: str) -> Path | None:
    """Resolve a dotted module name under sidekick package to its source file."""
    parts = module_name.split(".")
    if parts[0] == "sidekick":
        subpath = Path(*parts[1:])
        p1 = SIDEKICK_ROOT / (str(subpath) + ".py")
        if p1.is_file():
            return p1
        p2 = SIDEKICK_ROOT / subpath / "__init__.py"
        if p2.is_file():
            return p2
    return None


def get_ast_node_for_symbol(
    module_ast: ast.Module, symbol: str, module_path: Path
) -> tuple[ast.AST, Path] | None:
    """Find the AST definition node for *symbol* starting in *module_ast*."""
    # 1. Search locally in the module
    for node in module_ast.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == symbol:
                return node, module_path
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == symbol:
                    return node, module_path
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == symbol:
                return node, module_path

    # 2. Search import statements
    for node in module_ast.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name_in_module = alias.asname or alias.name
                if name_in_module == symbol:
                    imported_module = node.module
                    if imported_module:
                        if node.level > 0:
                            # Relative import
                            current_pkg = module_path.parent
                            for _ in range(node.level - 1):
                                current_pkg = current_pkg.parent
                            parts = imported_module.split(".")
                            imported_path = current_pkg / Path(*parts)
                            p1 = Path(str(imported_path) + ".py")
                            p2 = imported_path / "__init__.py"
                            resolved_file = (
                                p1 if p1.is_file() else (p2 if p2.is_file() else None)
                            )
                        else:
                            # Absolute import
                            resolved_file = resolve_module_file(imported_module)

                        if resolved_file:
                            with open(resolved_file, encoding="utf-8") as f:
                                child_ast = ast.parse(
                                    f.read(), filename=str(resolved_file)
                                )
                            return get_ast_node_for_symbol(
                                child_ast, alias.name, resolved_file
                            )
    return None


def extract_signature_from_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, Any]:
    """Extract arguments and return annotation from a function AST node."""
    args_info = []
    # Positional/keyword arguments
    for i, arg in enumerate(node.args.args):
        arg_name = arg.arg
        annotation = ast.unparse(arg.annotation) if arg.annotation else None

        default_val = None
        default_offset = len(node.args.args) - len(node.args.defaults)
        if i >= default_offset:
            default_val = ast.unparse(node.args.defaults[i - default_offset])

        args_info.append(
            {
                "name": arg_name,
                "annotation": annotation,
                "default": default_val,
                "kind": "arg",
            }
        )

    # vararg (*args)
    if node.args.vararg:
        args_info.append(
            {
                "name": node.args.vararg.arg,
                "annotation": (
                    ast.unparse(node.args.vararg.annotation)
                    if node.args.vararg.annotation
                    else None
                ),
                "default": None,
                "kind": "vararg",
            }
        )

    # keyword-only arguments
    for i, arg in enumerate(node.args.kwonlyargs):
        arg_name = arg.arg
        annotation = ast.unparse(arg.annotation) if arg.annotation else None

        default_val = None
        kw_default = node.args.kw_defaults[i]
        if kw_default is not None:
            default_val = ast.unparse(kw_default)

        args_info.append(
            {
                "name": arg_name,
                "annotation": annotation,
                "default": default_val,
                "kind": "kwonlyarg",
            }
        )

    # kwarg (**kwargs)
    if node.args.kwarg:
        args_info.append(
            {
                "name": node.args.kwarg.arg,
                "annotation": (
                    ast.unparse(node.args.kwarg.annotation)
                    if node.args.kwarg.annotation
                    else None
                ),
                "default": None,
                "kind": "kwarg",
            }
        )

    return {
        "args": args_info,
        "returns": ast.unparse(node.returns) if node.returns else None,
    }


def extract_class_info(node: ast.ClassDef, file_path: Path) -> dict[str, Any]:
    """Extract bases and public methods from a class AST node."""
    bases = [ast.unparse(base) for base in node.bases]
    methods = {}

    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method_name = item.name
            if method_name == "__init__" or not method_name.startswith("_"):
                methods[method_name] = extract_signature_from_function(item)

    return {"bases": bases, "methods": methods}


def extract_module_api(file_path: Path) -> dict[str, Any]:
    """Extract public symbols and their structure from a sidekick file."""
    with open(file_path, encoding="utf-8") as f:
        module_ast = ast.parse(f.read(), filename=str(file_path))

    all_symbols: list[str] = []
    for node in module_ast.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        all_symbols = [
                            ast.unparse(elt).strip("'\"") for elt in node.value.elts
                        ]

    api_info = {}
    for symbol in all_symbols:
        res = get_ast_node_for_symbol(module_ast, symbol, file_path)
        if res:
            sym_node, resolved_path = res
            if isinstance(sym_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                api_info[symbol] = {
                    "type": "function",
                    "signature": extract_signature_from_function(sym_node),
                }
            elif isinstance(sym_node, ast.ClassDef):
                api_info[symbol] = {
                    "type": "class",
                    "info": extract_class_info(sym_node, resolved_path),
                }
            elif isinstance(sym_node, (ast.Assign, ast.AnnAssign)):
                api_info[symbol] = {"type": "variable"}
        else:
            api_info[symbol] = {"type": "unknown"}

    return {"__all__": all_symbols, "symbols": api_info}


def test_sidekick_public_api_stability(pytestconfig: pytest.Config) -> None:
    """Assert that the public API signatures match sidekick_api_baseline.json."""
    current_api = {}
    for filename in TEST_FILES:
        path = SIDEKICK_ROOT / filename
        assert path.is_file(), f"Missing public sidekick module file: {filename}"
        current_api[filename] = extract_module_api(path)

    regenerate = pytestconfig.getoption("--regenerate-api-baseline")
    if regenerate:
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(current_api, f, indent=2)
        log.info("Regenerated public API baseline in %s", BASELINE_PATH)
        return

    assert (
        BASELINE_PATH.is_file()
    ), "Baseline file not found. Run with --regenerate-api-baseline to create it."

    with open(BASELINE_PATH, encoding="utf-8") as f:
        baseline_api = json.load(f)

    # Compare keys
    assert set(current_api.keys()) == set(
        baseline_api.keys()
    ), "Set of public sidekick module files changed."

    # Perform detailed comparison to raise clean assertions
    mismatches = []
    for filename, baseline_module in baseline_api.items():
        current_module = current_api[filename]

        if baseline_module["__all__"] != current_module["__all__"]:
            mismatches.append(
                f"{filename}: __all__ mismatch."
                f"\nExpected: {baseline_module['__all__']}"
                f"\nGot:      {current_module['__all__']}"
            )
            continue

        for symbol in baseline_module["__all__"]:
            if symbol not in current_module["symbols"]:
                mismatches.append(
                    f"{filename}: Symbol {symbol!r} missing from current API."
                )
                continue

            b_sym = baseline_module["symbols"][symbol]
            c_sym = current_module["symbols"][symbol]

            if b_sym != c_sym:
                mismatches.append(
                    f"{filename} / {symbol}: Signature changed."
                    f"\nExpected: {json.dumps(b_sym, indent=2)}"
                    f"\nGot:      {json.dumps(c_sym, indent=2)}"
                )

    if mismatches:
        pytest.fail(
            "Sidekick public API stability violation(s) detected:\n"
            + "\n".join(mismatches)
        )
