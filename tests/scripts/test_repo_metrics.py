"""Tests for the pure-Python repository metric counters.

These replace the shell pipelines (``grep -rnw ... | wc -l``,
``find ... -name ... | wc -l``, ``ls``) that
``generate_real_assessments.py`` previously ran through
``subprocess.check_output(..., shell=True)``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.repo_metrics import (
    count_files,
    count_matching_lines,
    list_directory_entries,
)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A small source tree exercising every counter."""
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "a.py").write_text(
        "# TODO: first\nx = 1  # TODO: second\n# TODONT is not a match\n",
        encoding="utf-8",
    )
    (src / "pkg" / "b.py").write_text("raise NotImplementedError\n", encoding="utf-8")
    (src / "widget.ts").write_text("// TODO: ts one\n", encoding="utf-8")
    (src / "widget.test.ts").write_text("", encoding="utf-8")
    (src / "view.test.tsx").write_text("", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_a.py").write_text("", encoding="utf-8")
    (tests / "helper.py").write_text("", encoding="utf-8")
    return tmp_path


class TestCountMatchingLines:
    def test_counts_lines_not_files_and_honours_word_boundaries(self, tree: Path):
        # Two TODO lines in a.py, one in widget.ts. "TODONT" must not match,
        # which is what `grep -w` gave us.
        assert count_matching_lines([tree / "src"], "TODO") == 3

    def test_counts_a_line_once_even_with_several_matches(self, tmp_path: Path):
        (tmp_path / "f.py").write_text("TODO TODO TODO\n", encoding="utf-8")
        assert count_matching_lines([tmp_path], "TODO") == 1

    def test_returns_zero_when_absent(self, tree: Path):
        assert count_matching_lines([tree / "src"], "FIXME") == 0

    def test_skips_unreadable_binary_content(self, tmp_path: Path):
        (tmp_path / "blob.bin").write_bytes(b"\xff\xfe\x00TODO\x00")
        (tmp_path / "real.py").write_text("# TODO\n", encoding="utf-8")
        assert count_matching_lines([tmp_path], "TODO") == 1

    def test_missing_root_is_not_an_error(self, tmp_path: Path):
        assert count_matching_lines([tmp_path / "nope"], "TODO") == 0

    def test_rejects_an_empty_word(self, tmp_path: Path):
        with pytest.raises(ValueError, match="non-empty"):
            count_matching_lines([tmp_path], "")


class TestCountFiles:
    def test_counts_across_several_roots_and_patterns(self, tree: Path):
        # Mirrors: find tests src -name 'test_*.py' -o -name '*.test.ts' \
        #                       -o -name '*.test.tsx'
        assert (
            count_files(
                [tree / "tests", tree / "src"],
                ["test_*.py", "*.test.ts", "*.test.tsx"],
            )
            == 3
        )

    def test_counts_a_file_once_when_two_patterns_match_it(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("", encoding="utf-8")
        assert count_files([tmp_path], ["*.py", "a.*"]) == 1

    def test_recurses(self, tree: Path):
        assert count_files([tree / "src"], ["*.py"]) == 2

    def test_ignores_directories_that_match_the_pattern(self, tmp_path: Path):
        (tmp_path / "bundle.py").mkdir()
        assert count_files([tmp_path], ["*.py"]) == 0

    def test_rejects_an_empty_pattern_list(self, tmp_path: Path):
        with pytest.raises(ValueError, match="at least one pattern"):
            count_files([tmp_path], [])


class TestListDirectoryEntries:
    def test_lists_names_sorted(self, tmp_path: Path):
        for name in ("b.yml", "a.yml"):
            (tmp_path / name).write_text("", encoding="utf-8")
        assert list_directory_entries(tmp_path) == ["a.yml", "b.yml"]

    def test_missing_directory_yields_nothing(self, tmp_path: Path):
        assert list_directory_entries(tmp_path / "nope") == []
