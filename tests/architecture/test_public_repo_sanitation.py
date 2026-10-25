"""Architecture gate: no developer-specific absolute paths in tracked public text."""

from __future__ import annotations

from pathlib import Path

from scripts import check_public_repo_sanitation as sanitation

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_builtin_patterns_flag_windows_and_posix_homes() -> None:
    assert sanitation.BUILTIN_PATTERNS[0][1].search(r"C:\Users\alice\file")
    assert sanitation.BUILTIN_PATTERNS[0][1].search(r"C:\\Users\\alice\\file")
    assert sanitation.BUILTIN_PATTERNS[0][1].search("C:/Users/alice/file")
    assert sanitation.BUILTIN_PATTERNS[1][1].search("/home/bob/staff-repos/Tools")


def test_scan_reports_violation_in_tmp_tree(tmp_path: Path) -> None:
    dirty = tmp_path / "src" / "example.py"
    dirty.parent.mkdir(parents=True)
    dirty.write_text('ROOT = "/home/alice/checkout"\n', encoding="utf-8")
    (tmp_path / ".git").mkdir()
    import subprocess

    subprocess.run(
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "src/example.py"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    hits = sanitation.scan(tmp_path)
    assert len(hits) == 1
    assert hits[0].pattern_id == "posix-user-home"


def test_real_repository_passes_sanitation_gate() -> None:
    assert sanitation.scan(REPO_ROOT) == []
