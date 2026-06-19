"""Tests for scripts/select_tests_for_changes.py (issue #3324).

Source-keyed test selection must map a changed *source* file to the existing
test directory that exercises it, so editing production code without touching
its test file no longer runs zero tests for that code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = REPO_ROOT / "scripts" / "select_tests_for_changes.py"

_spec = importlib.util.spec_from_file_location("select_tests_for_changes", _SCRIPT)
assert _spec is not None and _spec.loader is not None
select_tests_for_changes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(select_tests_for_changes)


@pytest.mark.unit
def test_calc_backend_source_change_selects_its_test_dir() -> None:
    # Acceptance criterion from #3324: a change under calc_backend/routers/
    # must select calc_backend tests even with no test file changed.
    targets = select_tests_for_changes.select_targets(
        ["src/shared/python/calc_backend/routers/wgs_reactor.py"]
    )
    assert "tests/shared/python/calc_backend" in targets


@pytest.mark.unit
def test_only_existing_targets_are_emitted() -> None:
    # A made-up package maps to no on-disk target, so nothing is emitted
    # (the pytest invocation must never fail on a missing path).
    targets = select_tests_for_changes.select_targets(
        ["src/shared/python/this_package_does_not_exist/foo.py"]
    )
    assert targets == []


@pytest.mark.unit
def test_non_source_and_non_python_paths_are_ignored() -> None:
    targets = select_tests_for_changes.select_targets(
        [
            "docs/readme.md",
            "tests/shared/python/calc_backend/test_x.py",  # a test file, not src
            "src/shared/python/calc_backend/routers/wgs_reactor.txt",  # not .py
        ]
    )
    # The changed *test* file path is handled by the existing changed-test lane,
    # not by this source-keyed mapper.
    assert targets == []


@pytest.mark.unit
def test_in_tree_test_paths_are_ignored_by_source_mapper() -> None:
    targets = select_tests_for_changes.select_targets(
        ["src/shared/python/sidekick/tests/test_json_io_boundary_3333.py"]
    )

    assert targets == []


@pytest.mark.unit
def test_vendored_source_roots_do_not_select_origin_repo_tests() -> None:
    targets = select_tests_for_changes.select_targets(
        [
            "src/movement_optimizer/launch_pyqt6.py",
            "src/movement_optimizer/gui_registration.py",
            "src/pendulum_simulator/src/double_pendulum_golf/physics.py",
        ]
    )

    assert targets == []


@pytest.mark.unit
def test_sidekick_process_calculator_source_change_selects_focused_tests() -> None:
    targets = select_tests_for_changes.select_targets(
        [
            "src/shared/python/sidekick/process_calculators/psa_package/psa_gui.py",
            "src/shared/python/sidekick/process_calculators/wgs_reactor_calculator.py",
        ]
    )

    assert targets == [
        "src/shared/python/sidekick/tests/process_calculators/test_psa_gui.py",
        "src/shared/python/sidekick/tests/process_calculators/test_wgs_reactor_calculator.py",
    ]
    assert "src/shared/python/sidekick/tests" not in targets


@pytest.mark.unit
def test_unknown_sidekick_process_calculator_falls_back_to_process_suite() -> None:
    targets = select_tests_for_changes.select_targets(
        [
            "src/shared/python/sidekick/process_calculators/acid_gas_dewpoint_calculator.py"
        ]
    )

    assert targets == ["src/shared/python/sidekick/tests/process_calculators"]


@pytest.mark.unit
def test_sidekick_data_processing_source_change_selects_focused_tests() -> None:
    targets = select_tests_for_changes.select_targets(
        [
            "src/shared/python/sidekick/data_processing/core.py",
            "src/shared/python/sidekick/data_processing/embedding.py",
        ]
    )

    assert targets == [
        "src/shared/python/sidekick/tests/test_data_processor_engine_errors.py",
        "tests/shared/python/sidekick/data_processing/test_io.py",
        "tests/test_data_processor_embedding.py",
        "tests/test_phase2_critical_bugs.py",
    ]
    assert "src/shared/python/sidekick/tests" not in targets
    assert "tests/shared/python/sidekick" not in targets


@pytest.mark.unit
def test_sidekick_theme_bridge_change_selects_focused_import_test() -> None:
    targets = select_tests_for_changes.select_targets(
        ["src/shared/python/sidekick/theme/__init__.py"]
    )

    assert targets == ["src/shared/python/sidekick/tests/test_theme_init.py"]
    assert "src/shared/python/sidekick/tests" not in targets
    assert "tests/shared/python/sidekick" not in targets


@pytest.mark.unit
def test_sidekick_agent_source_change_selects_focused_agent_tests() -> None:
    targets = select_tests_for_changes.select_targets(
        [
            "src/shared/python/sidekick/agent/__init__.py",
            "src/shared/python/sidekick/agent/action_service.py",
        ]
    )

    assert targets == ["tests/unit/sidekick/agent/test_action_service.py"]
    assert "src/shared/python/sidekick/tests" not in targets
    assert "tests/shared/python/sidekick" not in targets


@pytest.mark.unit
def test_sidekick_tools_sidebar_source_change_selects_focused_tests() -> None:
    targets = select_tests_for_changes.select_targets(
        [
            "src/shared/python/sidekick/ui/tools_sidebar/appearance.py",
            "src/shared/python/sidekick/ui/tools_sidebar/os_terminal.py",
            "src/shared/python/sidekick/ui/tools_sidebar/python_repl_tab.py",
            "src/shared/python/sidekick/ui/tools_sidebar/registry.py",
            "src/shared/python/sidekick/ui/tools_sidebar/runtime_tab_settings.py",
            "src/shared/python/sidekick/ui/tools_sidebar/workspace_tab.py",
            "src/shared/python/sidekick/ui/widgets/mixins/data_processor_ops.py",
        ]
    )

    assert targets == [
        "tests/shared/python/sidekick/ui/test_tools_sidebar_data_processor.py",
        "tests/shared/python/sidekick/ui/test_tools_sidebar_registry.py",
        "tests/test_data_processor_tab_export.py",
        "tests/unit/sidekick/test_appearance.py",
        "tests/unit/sidekick/test_os_terminal_widget.py",
        "tests/unit/sidekick/test_python_repl_widget.py",
        "tests/unit/sidekick/test_python_repl_widget_renamed.py",
        "tests/unit/sidekick/test_runtime_tab_settings.py",
        "tests/unit/sidekick/test_tab_context_menu.py",
        "tests/unit/sidekick/test_tab_definitions_alias_regression.py",
        "tests/unit/sidekick/test_workspace_registry_subscribe.py",
        "tests/unit/sidekick/test_workspace_tab_table.py",
    ]
    assert "src/shared/python/sidekick/tests" not in targets
    assert "tests/shared/python/sidekick" not in targets


@pytest.mark.unit
def test_vessel_drafter_contracts_source_change_selects_contract_tests() -> None:
    targets = select_tests_for_changes.select_targets(
        ["src/vessel_drafter/python/vessel_drafter/contracts.py"]
    )

    assert targets == [
        "tests/vessel_drafter/test_contracts_fallback.py",
        "tests/vessel_drafter/test_contracts_unified.py",
        "tests/vessel_drafter/test_vessel_drafter_contracts.py",
    ]
    assert "src/vessel_drafter/tests" not in targets


@pytest.mark.unit
def test_mr_kinematics_source_change_selects_focused_contract_tests() -> None:
    targets = select_tests_for_changes.select_targets(
        ["src/rotation_converter/_mr_kinematics.py"]
    )

    assert targets == ["tests/rotation_converter/test_mr_kinematics_contracts_3736.py"]
    assert "tests/rotation_converter" not in targets


@pytest.mark.unit
def test_tools_config_loader_source_change_selects_focused_tests() -> None:
    targets = select_tests_for_changes.select_targets(["src/tools/config_loader.py"])

    assert targets == ["tests/tools/test_config_loader.py"]
    assert "tests/tools" not in targets


@pytest.mark.unit
def test_output_is_sorted_and_deduplicated() -> None:
    targets = select_tests_for_changes.select_targets(
        [
            "src/shared/python/sidekick/a.py",
            "src/shared/python/sidekick/b.py",
        ]
    )
    assert targets == sorted(set(targets))
