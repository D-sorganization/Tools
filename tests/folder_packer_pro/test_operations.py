from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from folder_packer_pro import operations
from folder_packer_pro.operations import (
    PackOperationMixin,
    ScanPreviewMixin,
    UnpackOperationMixin,
)
from folder_packer_pro.pack_engine import PackResult, UnpackResult


class ValueWidget:
    def __init__(self, value: Any = "") -> None:
        self.value = value

    def get(self) -> Any:
        return self.value

    def set(self, value: Any) -> None:
        self.value = value


@dataclass
class Button:
    states: list[dict[str, str]] = field(default_factory=list)

    def configure(self, **kwargs: str) -> None:
        self.states.append(kwargs)


class TextWidget:
    def __init__(self) -> None:
        self.state = "normal"
        self.content = ""

    def configure(self, **kwargs: str) -> None:
        if "state" in kwargs:
            self.state = kwargs["state"]

    def delete(self, start: str, end: str) -> None:
        assert (start, end) == ("1.0", "end")
        self.content = ""

    def insert(self, index: str, text: str) -> None:
        assert index == "1.0"
        self.content = text + self.content


class Tree:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.deleted: tuple[Any, ...] = ()

    def get_children(self) -> tuple[str, ...]:
        return ("old",)

    def delete(self, *items: Any) -> None:
        self.deleted = items
        self.rows.clear()

    def insert(
        self,
        parent: str,
        index: str,
        *,
        text: str,
        values: tuple[str, str, str],
        tags: tuple[str, ...],
    ) -> None:
        self.rows.append(
            {
                "parent": parent,
                "index": index,
                "text": text,
                "values": values,
                "tags": tags,
            }
        )


class Root:
    def __init__(self) -> None:
        self.callbacks: list[tuple[int, Any]] = []

    def after(self, delay: int, callback: Any) -> None:
        self.callbacks.append((delay, callback))
        callback()


class ImmediateThread:
    def __init__(self, *, target: Any, daemon: bool) -> None:
        self.target = target
        self.daemon = daemon

    def start(self) -> None:
        self.target()


class ScanHarness(ScanPreviewMixin):
    def __init__(self, source: Path | str = "") -> None:
        self.pack_source_entry = ValueWidget(str(source))
        self.include_git_var = ValueWidget(False)
        self.exclude_patterns: set[str] = set()
        self.stats_text = TextWidget()
        self.preview_tree = Tree()
        self.root = Root()
        self.preview_updates = 0

    def _update_preview_tree(self) -> None:
        self.preview_updates += 1


class PackHarness(PackOperationMixin):
    def __init__(self, source: Path, output: Path) -> None:
        self.pack_source_entry = ValueWidget(str(source))
        self.pack_output_entry = ValueWidget(str(output))
        self.encrypt_var = ValueWidget(False)
        self.pack_password_entry = ValueWidget("")
        self.pack_password_confirm = ValueWidget("")
        self.cancel_operation = False
        self.pack_btn = Button()
        self.pack_cancel_btn = Button()
        self.pack_progress_var = ValueWidget(0)
        self.compression_var = ValueWidget("balanced")
        self.create_manifest_var = ValueWidget(True)
        self.include_git_var = ValueWidget(False)
        self.exclude_patterns: set[str] = set()
        self.root = Root()
        self.statuses: list[str] = []
        self.logs: list[tuple[str, str]] = []
        self.finished = 0

    def _update_pack_status(self, message: str) -> None:
        self.statuses.append(message)

    def _log_message(self, message: str, level: str = "info") -> None:
        self.logs.append((message, level))

    def _pack_finished(self) -> None:
        self.finished += 1


class UnpackHarness(UnpackOperationMixin):
    def __init__(self, package: Path, dest: Path) -> None:
        self.unpack_source_entry = ValueWidget(str(package))
        self.unpack_dest_entry = ValueWidget(str(dest))
        self.encrypted_var = ValueWidget(False)
        self.unpack_password_entry = ValueWidget("")
        self.unpack_btn = Button()
        self.unpack_cancel_btn = Button()
        self.unpack_progress_var = ValueWidget(0)
        self.root = Root()
        self.statuses: list[str] = []
        self.logs: list[tuple[str, str]] = []
        self.finished = 0
        self.package_info_text = TextWidget()

    def _update_unpack_status(self, message: str) -> None:
        self.statuses.append(message)

    def _log_message(self, message: str, level: str = "info") -> None:
        self.logs.append((message, level))

    def _unpack_finished(self) -> None:
        self.finished += 1


def test_display_stats_formats_summary_and_refreshes_preview() -> None:
    harness = ScanHarness()
    harness._display_stats(
        {
            "total_files": 4,
            "total_size": 2048,
            "excluded_files": 1,
            "file_types": {".py": 3, ".md": 1},
        }
    )

    assert "Total Files: 4" in harness.stats_text.content
    assert "Total Size: 2.00 KB" in harness.stats_text.content
    assert ".py" in harness.stats_text.content
    assert harness.stats_text.state == "disabled"
    assert harness.preview_updates == 1


def test_populate_tree_adds_relative_paths_with_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    file_path = source / "main.py"
    file_path.write_text("print('ok')\n", encoding="utf-8")

    harness = ScanHarness(source)
    harness._populate_tree([(file_path, file_path.stat())], source)

    assert harness.preview_tree.rows[0]["text"] == "main.py"
    assert harness.preview_tree.rows[0]["values"][1] == "Code"
    assert harness.preview_tree.rows[0]["tags"] == (str(file_path),)


def test_update_preview_tree_clears_existing_rows_and_populates_first_500(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(505):
        (source / f"file_{index}.txt").write_text("x", encoding="utf-8")

    harness = ScanHarness(source)
    monkeypatch.setattr(operations.threading, "Thread", ImmediateThread)
    ScanPreviewMixin._update_preview_tree(harness)

    assert harness.preview_tree.deleted == ("old",)
    assert len(harness.preview_tree.rows) == 500


def test_start_pack_validates_required_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        operations.messagebox,
        "showwarning",
        lambda title, message: warnings.append((title, message)),
    )
    harness = PackHarness(tmp_path, tmp_path / "out.fpp")
    harness.pack_source_entry = ValueWidget("")

    harness._start_pack()

    assert warnings == [("No Source", "Please select a source folder.")]


def test_start_pack_validates_password_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        operations.messagebox,
        "showwarning",
        lambda title, message: warnings.append((title, message)),
    )
    harness = PackHarness(tmp_path, tmp_path / "out.fpp")
    harness.encrypt_var = ValueWidget(True)
    harness.pack_password_entry = ValueWidget("secret")
    harness.pack_password_confirm = ValueWidget("different")

    harness._start_pack()

    assert warnings == [("Password Mismatch", "Passwords do not match.")]


def test_run_pack_logs_success_and_updates_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "out.fpp"
    output.write_bytes(b"archive")
    harness = PackHarness(source, output)
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        operations.messagebox,
        "showinfo",
        lambda title, message: messages.append((title, message)),
    )
    monkeypatch.setattr(
        operations,
        "collect_files",
        lambda *args, **kwargs: [source / "a.txt", source / "b.txt"],
    )

    def fake_pack_files(**kwargs: Any) -> PackResult:
        kwargs["progress_callback"]("a.txt", 1, 2)
        kwargs["progress_callback"]("b.txt", 2, 2)
        return PackResult(
            success=True,
            output_path=output,
            total_files=2,
            package_size=7,
            errors=["minor warning"],
        )

    monkeypatch.setattr(operations, "pack_files", fake_pack_files)

    harness._run_pack()

    assert harness.pack_progress_var.get() == 100.0
    assert ("minor warning", "error") in harness.logs
    assert any("Package created successfully" in log[0] for log in harness.logs)
    assert messages[0][0] == "Success"
    assert harness.finished == 1


def test_run_pack_reports_engine_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    harness = PackHarness(source, tmp_path / "out.fpp")
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(operations, "collect_files", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        operations,
        "pack_files",
        lambda **kwargs: PackResult(success=False, error="disk full"),
    )
    monkeypatch.setattr(
        operations.messagebox,
        "showerror",
        lambda title, message: errors.append((title, message)),
    )

    harness._run_pack()

    assert ("Pack operation failed: disk full", "error") in harness.logs
    assert errors == [("Error", "Pack failed:\n\ndisk full")]
    assert harness.finished == 1


def test_run_pack_cancelled_after_collection_skips_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    harness = PackHarness(source, tmp_path / "out.fpp")

    def cancel_after_collect(*args: Any, **kwargs: Any) -> list[Path]:
        harness.cancel_operation = True
        return [source / "a.txt"]

    monkeypatch.setattr(operations, "collect_files", cancel_after_collect)
    monkeypatch.setattr(
        operations,
        "pack_files",
        lambda **kwargs: pytest.fail("pack_files should not run after cancellation"),
    )

    harness._run_pack()

    assert ("Pack operation cancelled", "warning") in harness.logs
    assert harness.finished == 1


def test_run_unpack_logs_success_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "archive.fpp"
    package.write_bytes(b"archive")
    dest = tmp_path / "dest"
    harness = UnpackHarness(package, dest)
    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        operations.messagebox,
        "showinfo",
        lambda title, message: messages.append((title, message)),
    )

    def fake_unpack_files(**kwargs: Any) -> UnpackResult:
        kwargs["progress_callback"]("a.txt", 1, 2)
        kwargs["progress_callback"]("b.txt", 2, 2)
        return UnpackResult(
            success=True,
            dest_path=dest,
            total_files=2,
            errors=["minor extraction warning"],
        )

    monkeypatch.setattr(operations, "unpack_files", fake_unpack_files)

    harness._run_unpack()

    assert harness.unpack_progress_var.get() == 100.0
    assert ("minor extraction warning", "error") in harness.logs
    assert any("Package extracted successfully" in log[0] for log in harness.logs)
    assert messages[0][0] == "Success"
    assert harness.finished == 1


def test_inspect_package_displays_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "archive.fpp"
    package.write_bytes(b"archive")
    harness = UnpackHarness(package, tmp_path / "dest")
    monkeypatch.setattr(
        operations,
        "inspect_package",
        lambda path: {
            "file": Path(path).name,
            "size_formatted": "12.00 B",
            "encrypted": False,
            "metadata": {
                "created_at": "2026-06-01T00:00:00+00:00",
                "total_files": 3,
                "compression": "balanced",
            },
        },
    )

    harness._inspect_package()

    assert "File: archive.fpp" in harness.package_info_text.content
    assert "Encrypted: No" in harness.package_info_text.content
    assert "Total Files: 3" in harness.package_info_text.content
    assert harness.package_info_text.state == "disabled"


def test_start_unpack_validates_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        operations.messagebox,
        "showwarning",
        lambda title, message: warnings.append((title, message)),
    )
    harness = UnpackHarness(tmp_path / "archive.fpp", tmp_path / "dest")
    harness.unpack_dest_entry = ValueWidget("")

    harness._start_unpack()

    assert warnings == [("No Destination", "Please select a destination folder.")]


def test_scan_folder_reports_missing_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        operations.messagebox,
        "showerror",
        lambda title, message: errors.append((title, message)),
    )
    harness = ScanHarness(tmp_path / "missing")

    harness._scan_folder()

    assert errors == [("Error", "Source folder does not exist.")]


def test_update_preview_tree_ignores_missing_source(tmp_path: Path) -> None:
    harness = ScanHarness(tmp_path / "missing")

    ScanPreviewMixin._update_preview_tree(harness)

    assert harness.preview_tree.deleted == ("old",)
    assert harness.preview_tree.rows == []


def test_populate_tree_rejects_none() -> None:
    with pytest.raises(ValueError, match="files must be provided"):
        ScanHarness()._populate_tree(None, Path.cwd())  # type: ignore[arg-type]


def test_progress_callbacks_reject_missing_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    pack_harness = PackHarness(source, tmp_path / "out.fpp")
    monkeypatch.setattr(operations, "collect_files", lambda *args, **kwargs: [source])

    def pack_with_bad_progress(**kwargs: Any) -> PackResult:
        with pytest.raises(ValueError, match="filename must be provided"):
            kwargs["progress_callback"](None, 1, 1)
        return PackResult(success=True, package_size=0)

    monkeypatch.setattr(operations, "pack_files", pack_with_bad_progress)
    monkeypatch.setattr(operations.messagebox, "showinfo", lambda *args: None)

    pack_harness._run_pack()

    unpack_harness = UnpackHarness(tmp_path / "archive.fpp", tmp_path / "dest")

    def unpack_with_bad_progress(**kwargs: Any) -> UnpackResult:
        with pytest.raises(ValueError, match="filename must be provided"):
            kwargs["progress_callback"](None, 1, 1)
        return UnpackResult(success=True, total_files=0)

    monkeypatch.setattr(operations, "unpack_files", unpack_with_bad_progress)
    unpack_harness._run_unpack()

    assert pack_harness.finished == 1
    assert unpack_harness.finished == 1
