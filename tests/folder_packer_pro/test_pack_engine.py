from __future__ import annotations

import base64
import builtins
import gzip
import json
from pathlib import Path

import pytest

from folder_packer_pro import pack_engine
from folder_packer_pro.pack_engine import (
    UNSAFE_ARCHIVE_PATH_MESSAGE,
    collect_files,
    inspect_package,
    pack_files,
    unpack_files,
)


def _write_package(
    path: Path,
    files: dict[str, bytes],
    *,
    compress: bool = True,
) -> None:
    payload = {
        "files": {
            name: base64.b64encode(content).decode("utf-8")
            for name, content in files.items()
        },
        "metadata": {
            "compression": "balanced" if compress else "none",
            "encrypted": False,
            "total_files": len(files),
        },
    }
    data = json.dumps(payload).encode("utf-8")
    path.write_bytes(gzip.compress(data) if compress else data)


def test_collect_files_filters_excluded_directories_and_supports_cancel(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "keep.txt").write_text("keep", encoding="utf-8")
    (source / "debug.log").write_text("skip", encoding="utf-8")
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("skip", encoding="utf-8")
    (source / "node_modules").mkdir()
    (source / "node_modules" / "dep.js").write_text("skip", encoding="utf-8")

    files = collect_files(source, {"*.log", "node_modules"}, include_git=False)

    assert files == [source / "keep.txt"]
    assert collect_files(source, set(), cancel_check=lambda: True) == []


def test_pack_unpack_round_trip_manifest_and_progress(tmp_path: Path) -> None:
    source = tmp_path / "project"
    nested = source / "src"
    nested.mkdir(parents=True)
    (source / "README.md").write_text("# Project\n", encoding="utf-8")
    (nested / "main.py").write_text("print('hello')\n", encoding="utf-8")
    output = tmp_path / "project.fpp"
    dest = tmp_path / "unpacked"
    progress: list[tuple[str, int, int]] = []

    result = pack_files(
        source_path=source,
        output_path=output,
        files_to_pack=[source / "README.md", nested / "main.py"],
        compression="balanced",
        encrypt=False,
        create_manifest=True,
        progress_callback=lambda name, current, total: progress.append(
            (name, current, total)
        ),
    )

    assert result.success is True
    assert result.total_files == 2
    assert result.package_size == output.stat().st_size
    assert progress == [("README.md", 1, 2), ("main.py", 2, 2)]
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["files"] == ["README.md", str(Path("src") / "main.py")]

    unpack_progress: list[tuple[str, int, int]] = []
    unpacked = unpack_files(
        output,
        dest,
        progress_callback=lambda name, current, total: unpack_progress.append(
            (name, current, total)
        ),
    )

    assert unpacked.success is True
    assert unpacked.total_files == 2
    assert (dest / "README.md").read_text(encoding="utf-8") == "# Project\n"
    assert (dest / "src" / "main.py").read_text(encoding="utf-8") == (
        "print('hello')\n"
    )
    assert unpack_progress == [("README.md", 1, 2), ("main.py", 2, 2)]


@pytest.mark.parametrize(
    "archive_name",
    [
        "../escape.txt",
        "/absolute.txt",
        "C:/absolute.txt",
        r"C:\absolute.txt",
        "nested/../escape.txt",
        "",
    ],
)
def test_unpack_rejects_archive_paths_that_escape_destination(
    tmp_path: Path,
    archive_name: str,
) -> None:
    package = tmp_path / "malicious.fpp"
    dest = tmp_path / "dest"
    _write_package(package, {archive_name: b"owned"})

    result = unpack_files(package, dest)

    assert result.success is True
    assert result.total_files == 1
    assert len(result.errors) == 1
    assert UNSAFE_ARCHIVE_PATH_MESSAGE in result.errors[0]
    assert not (tmp_path / "escape.txt").exists()
    assert not (dest / "absolute.txt").exists()


def test_pack_reports_cancelled_before_writing_archive(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    file_path = source / "data.txt"
    file_path.write_text("payload", encoding="utf-8")
    output = tmp_path / "cancelled.fpp"

    result = pack_files(
        source,
        output,
        [file_path],
        cancel_check=lambda: True,
    )

    assert result.success is False
    assert result.error == "Operation cancelled"
    assert not output.exists()


def test_pack_continues_after_unreadable_file_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    good = source / "good.txt"
    bad = source / "bad.txt"
    good.write_text("good", encoding="utf-8")
    bad.write_text("bad", encoding="utf-8")
    real_open = builtins.open

    def flaky_open(path: Path, *args: object, **kwargs: object) -> object:
        if Path(path) == bad:
            raise OSError("cannot read")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(pack_engine, "open", flaky_open, raising=False)

    result = pack_files(source, tmp_path / "partial.fpp", [good, bad])

    assert result.success is True
    assert result.total_files == 2
    assert len(result.errors) == 1
    assert "cannot read" in result.errors[0]


@pytest.mark.parametrize("compress", [True, False])
def test_inspect_package_reads_unencrypted_metadata(
    tmp_path: Path,
    compress: bool,
) -> None:
    package = tmp_path / "archive.fpp"
    _write_package(package, {"file.txt": b"content"}, compress=compress)

    info = inspect_package(package)

    assert info["file"] == "archive.fpp"
    assert info["encrypted"] is False
    assert info["metadata"]["encrypted"] is False
    assert info["metadata"]["compression"] == ("balanced" if compress else "none")


def test_inspect_package_marks_encrypted_or_unreadable_payload(tmp_path: Path) -> None:
    package = tmp_path / "encrypted.fpp"
    package.write_bytes(b"not-json-or-gzip")

    info = inspect_package(package)

    assert info["encrypted"] is True
    assert info["metadata"] == {}
