"""Tests for the changed-test-file assertion quality gate."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "check_test_assertions.py"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("check_test_assertions", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_assert_statement_counts_as_behavioral_assertion() -> None:
    module = _load_module()

    assert module.has_behavioral_assertion("def test_value():\n    assert 2 + 2 == 4\n")


def test_pytest_raises_counts_as_exception_assertion() -> None:
    module = _load_module()
    source = (
        "import pytest\n\n"
        "def test_invalid_value():\n"
        "    with pytest.raises(ValueError):\n"
        "        int('not-an-int')\n"
    )

    assert module.has_behavioral_assertion(source)


def test_comments_and_strings_do_not_count_as_assertions() -> None:
    module = _load_module()
    source = (
        "def test_smoke():\n"
        "    marker = 'assert result and pytest.raises(ValueError)'\n"
        "    # assert this comment should not count\n"
        "    run_smoke(marker)\n"
    )

    assert not module.has_behavioral_assertion(source)


def test_assertion_light_changed_test_file_fails(tmp_path: Path) -> None:
    module = _load_module()
    test_file = _write(
        tmp_path / "tests" / "test_smoke.py",
        "def test_smoke():\n    build_widget()\n",
    )

    violations = module.check_test_files([test_file], allowlist_patterns=())

    assert violations == [test_file]


def test_allowlisted_fixture_file_without_assertions_passes(tmp_path: Path) -> None:
    module = _load_module()
    fixture_file = _write(
        tmp_path / "tests" / "helpers" / "test_data_fixture.py",
        "def make_payload():\n    return {'value': 1}\n",
    )

    violations = module.check_test_files(
        [fixture_file],
        allowlist_patterns=("tests/helpers/test_data_fixture.py",),
        root=tmp_path,
    )

    assert violations == []


def test_plot_definition_support_exemption_is_exact(tmp_path: Path) -> None:
    module = _load_module()
    patterns = module.load_allowlist(
        Path(__file__).resolve().parents[2] / "scripts" / "test_assertion_allowlist.txt"
    )
    support_file = _write(
        tmp_path
        / "tests"
        / "rate_of_closure"
        / "_variation_plot_definition_support.py",
        "def make_definition():\n    return {'schema': 2}\n",
    )
    real_test = _write(
        tmp_path / "tests" / "rate_of_closure" / "test_plot_definition.py",
        "def test_definition():\n    make_definition()\n",
    )

    violations = module.check_test_files(
        [support_file, real_test],
        allowlist_patterns=patterns,
        root=tmp_path,
    )

    assert violations == [real_test]


@pytest.mark.parametrize(
    "probe_name",
    [
        "pyqt_variation_render_probe.py",
        "pyqt_variation_visual_state_probe.py",
        "pyqt_visualization_tab_probe.py",
    ],
)
def test_pyqt_render_probe_exemption_is_exact(tmp_path: Path, probe_name: str) -> None:
    module = _load_module()
    patterns = module.load_allowlist(
        Path(__file__).resolve().parents[2] / "scripts" / "test_assertion_allowlist.txt"
    )
    render_probe = _write(
        tmp_path / "tests" / "rate_of_closure" / probe_name,
        "def main():\n    render_diagnostic_artifacts()\n",
    )
    adjacent_test = _write(
        tmp_path / "tests" / "rate_of_closure" / f"test_{probe_name}",
        "def test_render():\n    render_widget()\n",
    )

    violations = module.check_test_files(
        [render_probe, adjacent_test],
        allowlist_patterns=patterns,
        root=tmp_path,
    )

    assert violations == [adjacent_test]


def test_non_test_python_file_is_not_checked(tmp_path: Path) -> None:
    module = _load_module()
    source_file = _write(tmp_path / "src" / "feature.py", "def run():\n    pass\n")

    selected = module.select_python_test_files([source_file], root=tmp_path)

    assert selected == []
