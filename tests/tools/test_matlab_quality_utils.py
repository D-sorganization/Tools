"""TDD suite for matlab_quality_utils.py.

Tests cover MATLABQualityChecker initialization, static analysis helpers,
banned patterns, workspace pollution detection, and DbC violations.
"""

from pathlib import Path

import pytest

from src.shared.python.contracts import PreconditionError
from src.tools.matlab_quality_utils import MATLABQualityChecker

# ─── __init__ DbC ──────────────────────────────────────────────


def test_init_requires_path(tmp_path):
    """Non-Path raises PreconditionError."""
    with pytest.raises(PreconditionError):
        MATLABQualityChecker(str(tmp_path))  # type: ignore[arg-type]


def test_init_requires_absolute_path():
    """Relative Path raises PreconditionError."""
    with pytest.raises(PreconditionError):
        MATLABQualityChecker(Path("relative/path"))


def test_init_success(tmp_path):
    checker = MATLABQualityChecker(tmp_path)
    assert checker.project_root == tmp_path
    assert checker.matlab_dir == tmp_path / "matlab"


# ─── check_matlab_files_exist ──────────────────────────────────


def test_check_matlab_files_missing_dir(tmp_path):
    checker = MATLABQualityChecker(tmp_path)
    assert checker.check_matlab_files_exist() is False


def test_check_matlab_files_empty_dir(tmp_path):
    (tmp_path / "matlab").mkdir()
    checker = MATLABQualityChecker(tmp_path)
    assert checker.check_matlab_files_exist() is False


def test_check_matlab_files_found(tmp_path):
    matlab = tmp_path / "matlab"
    matlab.mkdir()
    (matlab / "script.m").write_text("% test")
    checker = MATLABQualityChecker(tmp_path)
    assert checker.check_matlab_files_exist() is True


# ─── _track_nesting ────────────────────────────────────────────


def test_track_nesting_enters_function():
    in_func, level = MATLABQualityChecker._track_nesting(
        "function y = foo(x)", False, 0
    )
    assert in_func is True
    assert level == 1


def test_track_nesting_exits_on_end():
    in_func, level = MATLABQualityChecker._track_nesting("end", True, 1)
    assert in_func is False
    assert level == 0


def test_track_nesting_nested_if():
    in_func, level = MATLABQualityChecker._track_nesting("if x > 0", True, 1)
    assert in_func is True
    assert level == 2


# ─── _check_banned_patterns ────────────────────────────────────


def test_check_banned_todo():
    issues: list[str] = []
    MATLABQualityChecker._check_banned_patterns(
        Path("script.m"), "% TODO: fix this", 5, issues
    )
    assert len(issues) == 1
    assert "TODO" in issues[0]


def test_check_banned_fixme():
    issues: list[str] = []
    MATLABQualityChecker._check_banned_patterns(
        Path("script.m"), "% FIXME: broken", 3, issues
    )
    assert any("FIXME" in i for i in issues)


def test_check_banned_clean_line():
    issues: list[str] = []
    MATLABQualityChecker._check_banned_patterns(
        Path("script.m"), "y = x + 1;", 1, issues
    )
    assert issues == []


def test_check_banned_dbc_non_path():
    issues: list[str] = []
    with pytest.raises(PreconditionError):
        MATLABQualityChecker._check_banned_patterns(
            "script.m",
            "% TODO",
            1,
            issues,  # type: ignore[arg-type]
        )


def test_check_banned_dbc_non_string_line():
    issues: list[str] = []
    with pytest.raises(PreconditionError):
        MATLABQualityChecker._check_banned_patterns(
            Path("f.m"),
            999,
            1,
            issues,  # type: ignore[arg-type]
        )


# ─── _check_workspace_pollution ────────────────────────────────


def test_workspace_pollution_clear_all_in_function():
    issues: list[str] = []
    MATLABQualityChecker._check_workspace_pollution(
        Path("f.m"), "clear all", 10, in_function=True, issues=issues
    )
    assert any("clear" in i.lower() for i in issues)


def test_workspace_pollution_clc_in_function():
    issues: list[str] = []
    MATLABQualityChecker._check_workspace_pollution(
        Path("f.m"), "clc", 5, in_function=True, issues=issues
    )
    assert any("clc" in i for i in issues)


def test_workspace_pollution_not_in_function_ok():
    issues: list[str] = []
    MATLABQualityChecker._check_workspace_pollution(
        Path("f.m"), "clear all", 1, in_function=False, issues=issues
    )
    assert issues == []


def test_workspace_pollution_dbc_non_path():
    issues: list[str] = []
    with pytest.raises(PreconditionError):
        MATLABQualityChecker._check_workspace_pollution(
            "f.m",
            "clc",
            1,
            True,
            issues,  # type: ignore[arg-type]
        )


# ─── _check_function_definition ────────────────────────────────


def test_check_function_def_no_docstring():
    lines = ["function y = foo(x)", "y = x + 1;", "end"]
    issues: list[str] = []
    MATLABQualityChecker._check_function_definition(Path("f.m"), lines, 1, issues)
    assert any("docstring" in i.lower() for i in issues)


def test_check_function_def_with_docstring():
    lines = [
        "function y = foo(x)",
        "% Computes the square of x.",
        "arguments",
        "    x (1,1) double",
        "end",
        "y = x^2;",
        "end",
    ]
    issues: list[str] = []
    MATLABQualityChecker._check_function_definition(Path("f.m"), lines, 1, issues)
    # Should still flag arguments block if present — but NOT docstring
    assert not any("docstring" in i.lower() for i in issues)


def test_check_function_def_dbc_non_list():
    with pytest.raises(PreconditionError):
        MATLABQualityChecker._check_function_definition(
            Path("f.m"),
            "not a list",
            1,
            [],  # type: ignore[arg-type]
        )


# ─── _static_matlab_analysis (integration) ─────────────────────


def test_static_analysis_no_issues(tmp_path):
    """Clean m-file with docstring and arguments produces no issues."""
    matlab = tmp_path / "matlab"
    matlab.mkdir()
    (matlab / "myFunc.m").write_text(
        "function y = myFunc(x)\n"
        "% Computes something useful.\n"
        "arguments\n"
        "    x (1,1) double\n"
        "end\n"
        "y = x * 2;\n"
        "end\n"
    )
    checker = MATLABQualityChecker(tmp_path)
    result = checker._static_matlab_analysis()
    assert result["success"] is True
    assert result["total_files"] == 1


def test_static_analysis_finds_todo(tmp_path):
    """m-file with TODO must produce at least one issue."""
    matlab = tmp_path / "matlab"
    matlab.mkdir()
    (matlab / "dirty.m").write_text("function y = foo(x)\n% TODO: fix\ny = x;\nend\n")
    checker = MATLABQualityChecker(tmp_path)
    result = checker._static_matlab_analysis()
    assert result["issues"]  # non-empty
