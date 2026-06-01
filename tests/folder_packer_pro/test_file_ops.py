from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from folder_packer_pro.file_ops import (
    collect_folder_stats,
    format_size,
    get_file_type,
    should_exclude,
)


def test_should_exclude_handles_git_wildcards_and_substrings(tmp_path: Path) -> None:
    assert should_exclude(tmp_path / ".git" / "config", set()) is True
    assert (
        should_exclude(tmp_path / ".git" / "config", set(), include_git=True) is False
    )
    assert should_exclude(tmp_path / "app.pyc", {"*.pyc"}) is True
    assert should_exclude(tmp_path / "node_modules", {"node_modules"}) is True
    assert should_exclude(tmp_path / "src" / "main.py", {"*.pyc"}) is False

    with pytest.raises(ValueError, match="path must be provided"):
        should_exclude(None, set())  # type: ignore[arg-type]


def test_collect_folder_stats_counts_types_sizes_and_exclusions(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (project / "README").write_text("readme", encoding="utf-8")
    (project / "debug.log").write_text("debug", encoding="utf-8")
    (project / ".git").mkdir()
    (project / ".git" / "config").write_text("git", encoding="utf-8")
    (project / "node_modules").mkdir()
    (project / "node_modules" / "dep.js").write_text("dep", encoding="utf-8")

    stats = collect_folder_stats(
        project,
        {"*.log", "node_modules"},
        include_git=False,
    )

    assert stats["total_files"] == 2
    assert (
        stats["total_size"]
        == (project / "main.py").stat().st_size + (project / "README").stat().st_size
    )
    assert stats["excluded_files"] == 1
    assert stats["file_types"] == defaultdict(int, {".py": 1, "no extension": 1})


def test_collect_folder_stats_allows_git_when_requested(tmp_path: Path) -> None:
    project = tmp_path / "project"
    git_dir = project / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "config").write_text("git", encoding="utf-8")

    stats = collect_folder_stats(project, set(), include_git=True)

    assert stats["total_files"] == 1
    assert stats["file_types"]["no extension"] == 1


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("module.py", "Code"),
        ("page.html", "Markup"),
        ("settings.toml", "Config"),
        ("logo.png", "Image"),
        ("song.flac", "Audio"),
        ("movie.mkv", "Video"),
        ("manual.pdf", "Document"),
        ("archive.bin", "Other"),
        ("UPPER.PY", "Code"),
    ],
)
def test_get_file_type_maps_known_extensions(filename: str, expected: str) -> None:
    assert get_file_type(Path(filename)) == expected


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, "0.00 B"),
        (1023, "1023.00 B"),
        (1024, "1.00 KB"),
        (1536, "1.50 KB"),
        (1024**2, "1.00 MB"),
        (1024**3, "1.00 GB"),
        (1024**4, "1.00 TB"),
        (1024**5, "1.00 PB"),
    ],
)
def test_format_size_uses_binary_units(size: int, expected: str) -> None:
    assert format_size(size) == expected
