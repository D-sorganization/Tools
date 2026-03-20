"""Unit tests for folder_packer_pro/file_ops.py.

Tests cover should_exclude, collect_folder_stats, get_file_type, and format_size.
All tests are headless-safe and use the tmp_path fixture for filesystem operations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from folder_packer_pro.file_ops import (
    collect_folder_stats,
    format_size,
    get_file_type,
    should_exclude,
)

# ---------------------------------------------------------------------------
# should_exclude
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestShouldExclude:
    """Tests for should_exclude()."""

    def test_excludes_git_directory_by_default(self, tmp_path: Path) -> None:
        git_path = tmp_path / ".git" / "config"
        assert should_exclude(git_path, set()) is True

    def test_includes_git_directory_when_flag_set(self, tmp_path: Path) -> None:
        git_path = tmp_path / ".git" / "config"
        assert should_exclude(git_path, set(), include_git=True) is False

    def test_excludes_by_glob_suffix_pattern(self, tmp_path: Path) -> None:
        path = tmp_path / "cache" / "module.pyc"
        assert should_exclude(path, {"*.pyc"}) is True

    def test_does_not_exclude_non_matching_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "module.py"
        assert should_exclude(path, {"*.pyc"}) is False

    def test_excludes_by_substring_pattern(self, tmp_path: Path) -> None:
        # Pattern is matched against path.name; here the directory itself is node_modules
        path = tmp_path / "node_modules"
        assert should_exclude(path, {"node_modules"}) is True

    def test_does_not_exclude_unrelated_path(self, tmp_path: Path) -> None:
        path = tmp_path / "src" / "main.py"
        assert should_exclude(path, {"node_modules", "*.pyc"}) is False

    def test_empty_exclude_patterns_no_exclusion(self, tmp_path: Path) -> None:
        path = tmp_path / "some_file.txt"
        assert should_exclude(path, set()) is False

    def test_precondition_path_none(self) -> None:
        with pytest.raises(AssertionError, match="path must be provided"):
            should_exclude(None, set())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# collect_folder_stats
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCollectFolderStats:
    """Tests for collect_folder_stats()."""

    def test_empty_folder(self, tmp_path: Path) -> None:
        stats = collect_folder_stats(tmp_path, set())
        assert stats["total_files"] == 0
        assert stats["total_size"] == 0
        assert stats["excluded_files"] == 0

    def test_counts_files_in_folder(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.py").write_text("world")
        stats = collect_folder_stats(tmp_path, set())
        assert stats["total_files"] == 2

    def test_accumulates_total_size(self, tmp_path: Path) -> None:
        content = "a" * 100
        (tmp_path / "file.txt").write_text(content)
        stats = collect_folder_stats(tmp_path, set())
        assert stats["total_size"] >= 100

    def test_tracks_file_types(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("code")
        (tmp_path / "readme.txt").write_text("docs")
        stats = collect_folder_stats(tmp_path, set())
        assert ".py" in stats["file_types"]
        assert ".txt" in stats["file_types"]

    def test_excluded_patterns_counted(self, tmp_path: Path) -> None:
        (tmp_path / "build.pyc").write_text("")
        (tmp_path / "main.py").write_text("code")
        stats = collect_folder_stats(tmp_path, {"*.pyc"})
        assert stats["excluded_files"] == 1
        assert stats["total_files"] == 1

    def test_excludes_git_by_default(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("git config")
        (tmp_path / "main.py").write_text("code")
        stats = collect_folder_stats(tmp_path, set(), include_git=False)
        assert stats["total_files"] == 1

    def test_includes_git_when_flag_set(self, tmp_path: Path) -> None:
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("git config")
        (tmp_path / "main.py").write_text("code")
        stats = collect_folder_stats(tmp_path, set(), include_git=True)
        assert stats["total_files"] == 2

    def test_file_with_no_extension(self, tmp_path: Path) -> None:
        (tmp_path / "Makefile").write_text("make")
        stats = collect_folder_stats(tmp_path, set())
        assert "no extension" in stats["file_types"]

    def test_precondition_folder_none(self) -> None:
        with pytest.raises(AssertionError, match="folder must be provided"):
            collect_folder_stats(None, set())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# get_file_type
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetFileType:
    """Tests for get_file_type()."""

    def test_python_is_code(self) -> None:
        assert get_file_type(Path("script.py")) == "Code"

    def test_javascript_is_code(self) -> None:
        assert get_file_type(Path("app.js")) == "Code"

    def test_html_is_markup(self) -> None:
        assert get_file_type(Path("index.html")) == "Markup"

    def test_css_is_markup(self) -> None:
        assert get_file_type(Path("style.css")) == "Markup"

    def test_json_is_config(self) -> None:
        assert get_file_type(Path("config.json")) == "Config"

    def test_yaml_is_config(self) -> None:
        assert get_file_type(Path("settings.yaml")) == "Config"

    def test_jpg_is_image(self) -> None:
        assert get_file_type(Path("photo.jpg")) == "Image"

    def test_png_is_image(self) -> None:
        assert get_file_type(Path("icon.png")) == "Image"

    def test_mp3_is_audio(self) -> None:
        assert get_file_type(Path("song.mp3")) == "Audio"

    def test_mp4_is_video(self) -> None:
        assert get_file_type(Path("clip.mp4")) == "Video"

    def test_pdf_is_document(self) -> None:
        assert get_file_type(Path("report.pdf")) == "Document"

    def test_txt_is_document(self) -> None:
        assert get_file_type(Path("notes.txt")) == "Document"

    def test_unknown_extension_is_other(self) -> None:
        assert get_file_type(Path("file.xyz")) == "Other"

    def test_uppercase_extension_normalized(self) -> None:
        # Extensions should be lowercased before comparison
        assert get_file_type(Path("SCRIPT.PY")) == "Code"

    def test_no_extension_is_other(self) -> None:
        assert get_file_type(Path("Makefile")) == "Other"


# ---------------------------------------------------------------------------
# format_size
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormatSize:
    """Tests for format_size()."""

    def test_bytes(self) -> None:
        result = format_size(512)
        assert "512" in result
        assert "B" in result

    def test_kilobytes(self) -> None:
        result = format_size(1024)
        assert "1.00 KB" == result

    def test_megabytes(self) -> None:
        result = format_size(1024 * 1024)
        assert "1.00 MB" == result

    def test_gigabytes(self) -> None:
        result = format_size(1024 * 1024 * 1024)
        assert "1.00 GB" == result

    def test_zero_bytes(self) -> None:
        result = format_size(0)
        assert "0.00 B" == result

    def test_fractional_kb(self) -> None:
        result = format_size(1536)  # 1.5 KB
        assert "1.50 KB" == result
