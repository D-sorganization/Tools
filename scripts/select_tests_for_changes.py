#!/usr/bin/env python3
"""Map changed source files to the test directories that exercise them.

Issue #3324: the PR ``tests`` lane selected tests only by *changed test files*,
so editing production code without touching its test file ran zero tests for
that code. This script closes that gap: it reads the list of changed Python
files (the ``changed_python_files.txt`` already produced by CI's "Collect
Changed Coverage Inputs" step) and prints the set of existing test targets that
cover the changed source packages.

The mapping is deliberately conservative — it only emits targets that actually
exist on disk, so the resulting ``pytest`` invocation never fails on a missing
path. Output is one target per line on stdout, suitable for ``mapfile``/``$()``
capture in a workflow step.

Usage::

    python scripts/select_tests_for_changes.py changed_python_files.txt
    python scripts/select_tests_for_changes.py < changed_python_files.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_SIDEKICK_PROCESS_CALCULATOR_TESTS = {
    "psa_package/psa_gui.py": [
        "src/shared/python/sidekick/tests/process_calculators/test_psa_gui.py",
    ],
    "psa_package/psa_model.py": [
        "src/shared/python/sidekick/tests/process_calculators/test_psa_model.py",
    ],
    "psa_package/psa_webapp.py": [
        "src/shared/python/sidekick/tests/process_calculators/test_psa_webapp.py",
    ],
    "wgs_reactor_calculator.py": [
        "src/shared/python/sidekick/tests/process_calculators/test_wgs_reactor_calculator.py",
    ],
}

_STANDALONE_TESTS = [
    "tests/unit/sidekick/test_standalone_public_api_baseline.py",
    "tests/unit/sidekick/test_standalone_runtime.py",
]
_PROFILE_RUNTIME_TESTS = [
    "tests/shared/python/sidekick/ui/test_tools_sidebar_state_profiles.py",
    *_STANDALONE_TESTS,
]

_SIDEKICK_SOURCE_TESTS = {
    **{
        source_path: _PROFILE_RUNTIME_TESTS
        for source_path in (
            "persistence/__init__.py",
            "persistence/schema.py",
            "persistence/state_profile.py",
        )
    },
    **{
        source_path: _STANDALONE_TESTS
        for source_path in (
            "__main__.py",
            "standalone/__init__.py",
            "standalone/onboarding.py",
            "standalone/preferences.py",
            "standalone/runner.py",
            "standalone/session_store.py",
            "standalone/window.py",
        )
    },
    "agent/__init__.py": [
        "tests/unit/sidekick/agent/test_action_service.py",
    ],
    "agent/action_service.py": [
        "tests/unit/sidekick/agent/test_action_service.py",
    ],
    "data_processing/core.py": [
        "src/shared/python/sidekick/tests/test_data_processor_engine_errors.py",
        "tests/shared/python/sidekick/data_processing/test_io.py",
        "tests/test_phase2_critical_bugs.py",
    ],
    "data_processing/embedding.py": [
        "tests/test_data_processor_embedding.py",
        "tests/test_phase2_critical_bugs.py",
    ],
    "theme/__init__.py": [
        "src/shared/python/sidekick/tests/test_theme_init.py",
    ],
    "ui/tools_sidebar/python_repl_tab.py": [
        "tests/unit/sidekick/test_python_repl_widget.py",
        "tests/unit/sidekick/test_python_repl_widget_renamed.py",
    ],
    "ui/tools_sidebar/appearance.py": [
        "tests/unit/sidekick/test_appearance.py",
    ],
    "ui/tools_sidebar/os_terminal.py": [
        "tests/unit/sidekick/test_os_terminal_widget.py",
        "tests/unit/sidekick/test_tab_context_menu.py",
    ],
    "ui/tools_sidebar/registry.py": [
        "tests/shared/python/sidekick/ui/test_tools_sidebar_registry.py",
        "tests/unit/sidekick/test_tab_context_menu.py",
        "tests/unit/sidekick/test_tab_definitions_alias_regression.py",
        "tests/unit/sidekick/test_workspace_registry_subscribe.py",
    ],
    "ui/tools_sidebar/runtime_tab_settings.py": [
        "tests/unit/sidekick/test_appearance.py",
        "tests/unit/sidekick/test_runtime_tab_settings.py",
        "tests/unit/sidekick/test_tab_definitions_alias_regression.py",
    ],
    "ui/tools_sidebar/workspace_tab.py": [
        "tests/unit/sidekick/test_workspace_tab_table.py",
    ],
    "ui/widgets/mixins/data_processor_ops.py": [
        "tests/shared/python/sidekick/ui/test_tools_sidebar_data_processor.py",
        "tests/test_data_processor_tab_export.py",
    ],
}

_VESSEL_DRAFTER_SOURCE_TESTS = {
    "python/vessel_drafter/contracts.py": [
        "tests/vessel_drafter/test_contracts_fallback.py",
        "tests/vessel_drafter/test_contracts_unified.py",
        "tests/vessel_drafter/test_vessel_drafter_contracts.py",
    ],
}

_TOP_LEVEL_SOURCE_TESTS = {
    ("rotation_converter", "_mr_kinematics.py"): [
        "tests/rotation_converter/test_mr_kinematics_contracts_3736.py",
    ],
    ("rotation_converter", "_mr_dynamics.py"): [
        "tests/rotation_converter/test_modern_robotics.py",
    ],
    ("rotation_converter", "modern_robotics.py"): [
        "tests/rotation_converter/test_modern_robotics.py",
    ],
    ("rotation_converter", "modern_robotics_pkg/dynamics.py"): [
        "tests/rotation_converter/test_modern_robotics.py",
    ],
    ("rotation_converter", "modern_robotics_pkg/kinematics.py"): [
        "tests/rotation_converter/test_modern_robotics.py",
    ],
    ("rotation_converter", "modern_robotics_pkg/trajectory.py"): [
        "tests/rotation_converter/test_modern_robotics.py",
    ],
    ("tools", "config_loader.py"): [
        "tests/tools/test_config_loader.py",
    ],
}

_VENDORED_SOURCE_PREFIXES = (
    ("src", "movement_optimizer"),
    ("src", "pendulum_simulator"),
)


def _read_changed_files(argv: list[str]) -> list[str]:
    """Read changed-file paths from a path argument or stdin."""
    if len(argv) > 1 and argv[1] not in {"-", ""}:
        text = Path(argv[1]).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()
    return [line.strip() for line in text.splitlines() if line.strip()]


def _candidate_targets(src_path: str) -> list[Path]:
    """Return candidate test targets for a single changed source path.

    Translates ``src/<package>/...`` and ``src/shared/python/<package>/...``
    source paths to the ``tests/<package>/`` and in-tree ``src/**/tests/``
    directories that conventionally mirror them.
    """
    p = Path(src_path)
    parts = p.parts
    targets: list[Path] = []

    if parts[:5] == ("src", "shared", "python", "sidekick", "process_calculators"):
        rel_process_path = Path(*parts[5:]).as_posix()
        for test_path in _SIDEKICK_PROCESS_CALCULATOR_TESTS.get(
            rel_process_path,
            ["src/shared/python/sidekick/tests/process_calculators"],
        ):
            targets.append(REPO_ROOT / test_path)
        return targets

    if parts[:4] == ("src", "shared", "python", "codemap"):
        targets.append(REPO_ROOT / "tests" / "unit" / "codemap")
        return targets

    if parts[:4] == ("src", "shared", "python", "sidekick"):
        rel_sidekick_path = Path(*parts[4:]).as_posix()
        if rel_sidekick_path in _SIDEKICK_SOURCE_TESTS:
            for test_path in _SIDEKICK_SOURCE_TESTS[rel_sidekick_path]:
                targets.append(REPO_ROOT / test_path)
            return targets

    if parts[:2] == ("src", "vessel_drafter"):
        rel_vessel_path = Path(*parts[2:]).as_posix()
        if rel_vessel_path in _VESSEL_DRAFTER_SOURCE_TESTS:
            for test_path in _VESSEL_DRAFTER_SOURCE_TESTS[rel_vessel_path]:
                targets.append(REPO_ROOT / test_path)
            return targets

    if len(parts) >= 3 and parts[0] == "src":
        exact_top_level_tests = _TOP_LEVEL_SOURCE_TESTS.get(
            (parts[1], Path(*parts[2:]).as_posix())
        )
        if exact_top_level_tests is not None:
            for test_path in exact_top_level_tests:
                targets.append(REPO_ROOT / test_path)
            return targets

    # Identify the changed file's *package root* so we can mirror it. For a
    # shared-library path the package is its 4th segment; for a top-level tool
    # it is the 2nd. We never walk above that root, so an unrelated ``tests``
    # dir further up (e.g. src/shared/python/tests) does not match every change.
    if parts[:3] == ("src", "shared", "python") and len(parts) >= 4:
        package_depth = 4
        mirror = REPO_ROOT / "tests" / "shared" / "python" / parts[3]
    elif parts and parts[0] == "src" and len(parts) >= 2:
        package_depth = 2
        mirror = REPO_ROOT / "tests" / parts[1]
    else:
        return targets

    # In-tree test package: only consider a ``tests`` dir at or below the
    # package root (inclusive), walking up no further than the package itself.
    for i in range(len(parts) - 1, package_depth - 1, -1):
        maybe = REPO_ROOT.joinpath(*parts[:i], "tests")
        if maybe.is_dir():
            targets.append(maybe)
            break

    # Mirrored top-level tests/<package>/ directory.
    targets.append(mirror)

    return targets


def select_targets(changed_files: list[str]) -> list[str]:
    """Return sorted, de-duplicated, existing test targets for changed sources."""
    selected: set[str] = set()
    for src_path in changed_files:
        # Only source files drive source-keyed selection; changed *test* files
        # are already handled by the existing changed_test_files.txt path.
        if not src_path.startswith("src/") or not src_path.endswith(".py"):
            continue
        parts = Path(src_path).parts
        if "tests" in parts:
            continue
        if any(parts[: len(prefix)] == prefix for prefix in _VENDORED_SOURCE_PREFIXES):
            continue
        for target in _candidate_targets(src_path):
            if target.exists():
                selected.add(target.relative_to(REPO_ROOT).as_posix())
    return sorted(selected)


def main(argv: list[str]) -> int:
    changed = _read_changed_files(argv)
    for target in select_targets(changed):
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
