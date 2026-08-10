"""Python-version compatibility contracts for Rate of Closure modules."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
STRING_ENUM_MODULES = (
    Path("src/rate_of_closure/simulation/manual_delivery.py"),
    Path("src/shared/python/swing_sim/conventions/registry.py"),
    Path("src/shared/python/swing_sim/flight/capability_contract.py"),
    Path("src/shared/python/swing_sim/flight/impact_solution_contract.py"),
    Path("src/shared/python/swing_sim/flight/inverse_contract.py"),
    Path("src/shared/python/swing_sim/flight/result_contract.py"),
    Path("src/shared/python/swing_sim/impact/dplane.py"),
)
UTC_MODULES = (Path("src/rate_of_closure/ui/pyqt6/torque_profile_controller.py"),)


@pytest.mark.parametrize("relative_path", STRING_ENUM_MODULES)
def test_string_enums_use_the_shared_python310_compatibility_contract(
    relative_path: Path,
) -> None:
    """Prevent newer interpreters from masking direct ``enum.StrEnum`` imports."""
    source_path = REPOSITORY_ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=source_path)
    runtime_nodes: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name):
            if node.test.id == "TYPE_CHECKING":
                runtime_nodes.extend(node.orelse)
                continue
        runtime_nodes.append(node)
    str_enum_imports = {
        node.module
        for runtime_node in runtime_nodes
        for node in ast.walk(runtime_node)
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == "StrEnum" for alias in node.names)
    }

    assert str_enum_imports == {"shared.python.compatibility"}


@pytest.mark.parametrize("relative_path", UTC_MODULES)
def test_utc_uses_the_shared_python310_compatibility_contract(
    relative_path: Path,
) -> None:
    """Prevent a direct Python 3.11-only ``datetime.UTC`` runtime import."""
    source_path = REPOSITORY_ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=source_path)
    utc_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and any(alias.name == "UTC" for alias in node.names)
    }

    assert utc_imports == {"shared.python.compatibility"}
