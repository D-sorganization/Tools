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
def test_output_is_sorted_and_deduplicated() -> None:
    targets = select_tests_for_changes.select_targets(
        [
            "src/shared/python/sidekick/a.py",
            "src/shared/python/sidekick/b.py",
        ]
    )
    assert targets == sorted(set(targets))
