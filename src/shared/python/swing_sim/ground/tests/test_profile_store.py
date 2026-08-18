"""Atomic persistence tests for strict ground profile libraries."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import shared.python.swing_sim.ground.profile_store as profile_store_module
import shared.python.swing_sim.ground.profile_store_platform as platform_module
from shared.python.swing_sim.ground.profile_store import (
    GroundProfileLibraryStore,
    ProfileStoreConflictError,
    ProfileStoreCorruptionError,
    ProfileStoreIndeterminateCommitError,
    ProfileStoreLockError,
    ProfileStorePathError,
    StoredGroundProfileLibrary,
)
from shared.python.swing_sim.ground.profile_types import GroundProfileLibrary

from .test_profile_contract import _library


def _changed_library() -> GroundProfileLibrary:
    library = _library()
    return replace(library, revision="1.0.1")


def test_store_create_load_compare_and_swap_and_backup(tmp_path: Path) -> None:
    store = GroundProfileLibraryStore(tmp_path)
    initial = store.save(_library(), expected_sha256=None)

    assert initial.library == _library()
    assert initial.sha256 == _library().canonical_sha256()
    assert initial.path == store.path
    assert store.load() == initial
    assert store.path.read_text(encoding="utf-8") == _library().to_json()

    with pytest.raises(ProfileStoreConflictError, match="already exists"):
        store.save(_library(), expected_sha256=None)
    with pytest.raises(ProfileStoreConflictError, match="digest"):
        store.save(_changed_library(), expected_sha256="0" * 64)

    changed = store.save(_changed_library(), expected_sha256=initial.sha256)
    assert changed.library == _changed_library()
    assert store.load() == changed
    assert store.load_backup().library == _library()
    assert not tuple(tmp_path.glob("*.tmp"))


def test_corrupt_primary_never_auto_recovers_and_recovery_is_explicit(
    tmp_path: Path,
) -> None:
    store = GroundProfileLibraryStore(tmp_path)
    initial = store.save(_library(), expected_sha256=None)
    store.save(_changed_library(), expected_sha256=initial.sha256)
    store.path.write_text("{corrupt", encoding="utf-8")
    corrupt_digest = hashlib.sha256(store.path.read_bytes()).hexdigest()

    with pytest.raises(ProfileStoreCorruptionError, match="invalid"):
        store.load()
    assert store.path.read_text(encoding="utf-8") == "{corrupt"
    assert store.load_backup().library == _library()

    with pytest.raises(ProfileStoreConflictError, match="digest"):
        store.recover_from_backup(
            expected_primary_sha256="f" * 64,
            expected_backup_sha256=store.load_backup().sha256,
        )
    recovered = store.recover_from_backup(
        expected_primary_sha256=corrupt_digest,
        expected_backup_sha256=store.load_backup().sha256,
    )
    assert recovered.library == _library()
    assert store.load() == recovered


def test_store_rejects_oversize_noncanonical_symlink_and_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = GroundProfileLibraryStore(tmp_path, max_bytes=512)
    store.path.write_bytes(b"{" + b"x" * 512)
    with pytest.raises(ProfileStoreCorruptionError, match="size limit"):
        store.load()

    store.path.unlink()
    store = GroundProfileLibraryStore(tmp_path)
    store.path.write_text(_library().to_json() + "\n", encoding="utf-8")
    with pytest.raises(ProfileStoreCorruptionError, match="invalid"):
        store.load()

    store.path.unlink()
    store.lock_path.write_text("occupied", encoding="utf-8")
    with pytest.raises(ProfileStoreLockError, match="locked"):
        store.save(_library(), expected_sha256=None)

    store.lock_path.unlink()
    target = tmp_path / "outside.json"
    target.write_text(_library().to_json(), encoding="utf-8")
    try:
        store.path.symlink_to(target)
    except (OSError, NotImplementedError):
        monkeypatch.setattr(
            GroundProfileLibraryStore,
            "_is_link_like",
            staticmethod(lambda candidate: candidate == store.path),
        )
    with pytest.raises(ProfileStorePathError, match="reparse point"):
        store.load()


def test_failed_primary_replace_preserves_old_file_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = GroundProfileLibraryStore(tmp_path)
    initial = store.save(_library(), expected_sha256=None)
    original_replace = store._replace_atomic

    def fail_primary_replace(source: Path, destination: Path) -> None:
        if destination == store.path:
            raise OSError("injected primary replace failure")
        original_replace(source, destination)

    monkeypatch.setattr(store, "_replace_atomic", fail_primary_replace)
    with pytest.raises(ProfileStoreCorruptionError, match="atomic write failed"):
        store.save(_changed_library(), expected_sha256=initial.sha256)

    assert store.load().library == _library()
    assert not tuple(tmp_path.glob("*.tmp"))
    assert not store.lock_path.exists()


def test_store_requires_absolute_real_directory_and_safe_filename(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProfileStorePathError, match="absolute"):
        GroundProfileLibraryStore(Path("relative"))
    with pytest.raises(ProfileStorePathError, match="existing directory"):
        GroundProfileLibraryStore(tmp_path / "missing")
    with pytest.raises(ProfileStorePathError, match="plain filename"):
        GroundProfileLibraryStore(tmp_path, filename="../escape.json")
    for filename in (
        "file:stream",
        "CON",
        "con.json",
        "CON .txt",
        "COM¹.json",
        "a.",
        "a ",
        "bad?.json",
        "control\x01.json",
    ):
        with pytest.raises(ProfileStorePathError, match="safe plain filename"):
            GroundProfileLibraryStore(tmp_path, filename=filename)


def test_store_revalidates_root_identity_and_types_setup_io_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "store"
    root.mkdir()
    store = GroundProfileLibraryStore(root)
    store.save(_library(), expected_sha256=None)
    relocated = tmp_path / "relocated"
    root.rename(relocated)
    root.mkdir()

    with pytest.raises(ProfileStorePathError, match="identity"):
        store.load()

    fresh = GroundProfileLibraryStore(root)

    def fail_tempfile(*args: object, **kwargs: object) -> tuple[int, str]:
        raise OSError("injected mkstemp failure")

    monkeypatch.setattr(profile_store_module.tempfile, "mkstemp", fail_tempfile)
    with pytest.raises(ProfileStoreCorruptionError, match="atomic write failed"):
        fresh.save(_library(), expected_sha256=None)
    assert not fresh.lock_path.exists()


def test_recovery_compare_and_swap_binds_selected_backup(tmp_path: Path) -> None:
    store = GroundProfileLibraryStore(tmp_path)
    initial = store.save(_library(), expected_sha256=None)
    store.save(_changed_library(), expected_sha256=initial.sha256)
    store.path.write_text("{corrupt", encoding="utf-8")
    primary_digest = hashlib.sha256(store.path.read_bytes()).hexdigest()

    with pytest.raises(ProfileStoreConflictError, match="backup.*digest"):
        store.recover_from_backup(
            expected_primary_sha256=primary_digest,
            expected_backup_sha256="f" * 64,
        )


def test_post_replace_sync_failure_reports_indeterminate_committed_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = GroundProfileLibraryStore(tmp_path)

    def fail_sync() -> None:
        raise OSError("injected directory sync failure")

    monkeypatch.setattr(store, "_sync_directory", fail_sync)
    with pytest.raises(ProfileStoreIndeterminateCommitError) as failure:
        store.save(_library(), expected_sha256=None)

    assert failure.value.destination == store.path
    assert failure.value.committed_sha256 == _library().canonical_sha256()
    assert store.load().library == _library()


def test_stored_library_rejects_forged_digest_and_relative_path(
    tmp_path: Path,
) -> None:
    stored = GroundProfileLibraryStore(tmp_path).save(_library(), expected_sha256=None)

    with pytest.raises(ValueError, match="sha256"):
        replace(stored, sha256="0" * 64)
    with pytest.raises(ProfileStorePathError, match="absolute"):
        StoredGroundProfileLibrary(stored.library, stored.sha256, Path("relative"))


def test_windows_reparse_attribute_is_treated_as_link_like(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reparse = int(profile_store_module.stat.FILE_ATTRIBUTE_REPARSE_POINT)
    monkeypatch.setattr(
        profile_store_module.os,
        "lstat",
        lambda _path: SimpleNamespace(st_mode=0, st_file_attributes=reparse),
    )

    assert GroundProfileLibraryStore._is_link_like(tmp_path)


def test_windows_atomic_replace_uses_wide_write_through_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, int]] = []

    class MoveFile:
        argtypes: object = None
        restype: object = None

        def __call__(self, source: str, destination: str, flags: int) -> int:
            calls.append((source, destination, flags))
            return 1

    move_file = MoveFile()
    kernel32 = SimpleNamespace(MoveFileExW=move_file)
    monkeypatch.setattr(
        platform_module.ctypes,
        "WinDLL",
        lambda *_args, **_kwargs: kernel32,
        raising=False,
    )

    source = tmp_path / "source"
    destination = tmp_path / "destination"
    platform_module._windows_atomic_replace(source, destination)

    assert calls == [(str(source), str(destination), 0x1 | 0x8)]
    assert move_file.argtypes == (
        platform_module.ctypes.c_wchar_p,
        platform_module.ctypes.c_wchar_p,
        platform_module.ctypes.c_uint32,
    )
    assert move_file.restype is platform_module.ctypes.c_int


def test_root_reparse_probe_failures_are_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = GroundProfileLibraryStore(tmp_path)

    def fail_probe(_path: Path) -> bool:
        raise OSError("injected root probe failure")

    monkeypatch.setattr(
        GroundProfileLibraryStore,
        "_is_link_like",
        staticmethod(fail_probe),
    )

    with pytest.raises(ProfileStorePathError, match="identity"):
        store.load()
