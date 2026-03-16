"""Comprehensive TDD suite for quality_utils.py.

Tests cover all public functions with valid inputs, edge cases,
and precondition violations (DbC contracts).
"""

from pathlib import Path

import pytest

from src.shared.python.contracts import PreconditionError
from tools.quality_utils import (
    check_ast_issues,
    check_banned_patterns,
    check_file,
    check_magic_numbers,
    is_legitimate_pass_context,
    is_legitimate_tkinter_binding,
    strip_comments_from_line,
)

# ─── is_legitimate_pass_context ────────────────────────────────


def test_pass_in_class_is_legitimate():
    lines = ["class Foo:", "    pass"]
    assert is_legitimate_pass_context(lines, 2) is True


def test_pass_in_try_is_legitimate():
    lines = ["try:", "    pass"]
    assert is_legitimate_pass_context(lines, 2) is True


def test_pass_after_def_is_not_legitimate():
    lines = ["def foo():", "    pass"]
    assert is_legitimate_pass_context(lines, 2) is False


def test_pass_out_of_bounds_returns_false():
    lines = ["x = 1"]
    assert is_legitimate_pass_context(lines, 0) is False
    assert is_legitimate_pass_context(lines, 99) is False


def test_pass_context_dbc_rejects_bad_args():
    with pytest.raises(PreconditionError):
        is_legitimate_pass_context("not a list", 1)  # type: ignore[arg-type]
    with pytest.raises(PreconditionError):
        is_legitimate_pass_context([], "1")  # type: ignore[arg-type]


# ─── is_legitimate_tkinter_binding ─────────────────────────────


def test_tkinter_keyrelease_recognized():
    assert is_legitimate_tkinter_binding('root.bind("<KeyRelease>", handler)') is True


def test_tkinter_configure_recognized():
    assert (
        is_legitimate_tkinter_binding('widget.bind("<Configure>", on_resize)') is True
    )


def test_non_tkinter_line_not_legitimate():
    assert is_legitimate_tkinter_binding("x = 1 + 2") is False


def test_tkinter_binding_dbc_rejects_non_string():
    with pytest.raises(PreconditionError):
        is_legitimate_tkinter_binding(123)  # type: ignore[arg-type]


# ─── strip_comments_from_line ───────────────────────────────────


def test_strip_inline_comment():
    result = strip_comments_from_line("x = 1  # this is a comment")
    assert result == "x = 1"


def test_strip_does_not_strip_hash_in_string():
    result = strip_comments_from_line('msg = "hello # world"')
    assert "#" in result  # hash inside string must be preserved


def test_strip_full_comment_line():
    result = strip_comments_from_line("# full comment")
    assert result == ""


def test_strip_no_comment():
    result = strip_comments_from_line("x = 1 + 2")
    assert result == "x = 1 + 2"


def test_strip_dbc_rejects_non_string():
    with pytest.raises(PreconditionError):
        strip_comments_from_line(None)  # type: ignore[arg-type]


# ─── check_banned_patterns ─────────────────────────────────────


def test_check_banned_finds_todo():
    lines = ["# TODO: fix this eventually"]
    issues = check_banned_patterns(lines, Path("myfile.py"))
    assert len(issues) == 1
    assert "TODO" in issues[0][1]


def test_check_banned_finds_ellipsis():
    lines = ["def foo():", "    ..."]
    issues = check_banned_patterns(lines, Path("myfile.py"))
    assert any("Ellipsis" in i[1] for i in issues)


def test_check_banned_skips_quality_utils_itself():
    lines = ["# TODO: something"]
    issues = check_banned_patterns(lines, Path("quality_utils.py"))
    assert issues == []


def test_check_banned_dbc_rejects_bad_types():
    with pytest.raises(PreconditionError):
        check_banned_patterns("not a list", Path("f.py"))  # type: ignore[arg-type]
    with pytest.raises(PreconditionError):
        check_banned_patterns([], "not_a_path")  # type: ignore[arg-type]


# ─── check_magic_numbers ────────────────────────────────────────


def test_check_magic_numbers_finds_pi():
    lines = ["area = 3.141 * r * r"]
    issues = check_magic_numbers(lines, Path("calc.py"))
    assert len(issues) == 1
    assert "math.pi" in issues[0][1]


def test_check_magic_numbers_ignores_comment():
    lines = ["x = 1  # gravity is 9.8 m/s2"]
    issues = check_magic_numbers(lines, Path("calc.py"))
    assert issues == []


def test_check_magic_numbers_skips_excluded():
    lines = ["x = 3.141"]
    issues = check_magic_numbers(lines, Path("quality_utils.py"))
    assert issues == []


def test_check_magic_dbc_rejects_bad_types():
    with pytest.raises(PreconditionError):
        check_magic_numbers("not a list", Path("f.py"))  # type: ignore[arg-type]


# ─── check_ast_issues ───────────────────────────────────────────


def test_check_ast_finds_missing_docstring():
    content = "def my_func():\n    return 1"
    issues = check_ast_issues(content, Path("f.py"))
    assert any("missing docstring" in i[1] for i in issues)


def test_check_ast_no_issues_with_docstring():
    content = 'def my_func():\n    """A proper docstring."""\n    return 1'
    issues = check_ast_issues(content, Path("f.py"))
    assert issues == []


def test_check_ast_handles_syntax_error():
    content = "def :"  # broken syntax
    issues = check_ast_issues(content, Path("f.py"))
    assert any("Syntax error" in i[1] for i in issues)


# ─── check_file ─────────────────────────────────────────────────


def test_check_file_clean(tmp_path):
    f = tmp_path / "clean.py"
    f.write_text('def foo():\n    """Docstring."""\n    return 1\n', encoding="utf-8")
    issues = check_file(f)
    assert issues == []


def test_check_file_with_todo(tmp_path):
    f = tmp_path / "dirty.py"
    f.write_text("# TODO: clean me up\n", encoding="utf-8")
    issues = check_file(f)
    assert len(issues) >= 1


def test_check_file_dbc_rejects_missing_file(tmp_path):
    with pytest.raises(PreconditionError):
        check_file(tmp_path / "nonexistent.py")


def test_check_file_dbc_rejects_directory(tmp_path):
    with pytest.raises(PreconditionError):
        check_file(tmp_path)  # directory, not a file
