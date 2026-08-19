"""Focused contract tests for private authority-state filesystem handling."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

if os.name == "nt":
    import pywintypes
    import win32security

from rate_of_closure.web_authority.state_security import (
    PathInspection,
    PathKind,
    StateSecurityCode,
    StateSecurityError,
    bounded_state_path,
    create_private_state_root,
    prepare_private_state_root,
    verify_state_file,
    verify_state_root,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


class _Backend:
    """Deterministic test double around the host temporary filesystem."""

    def __init__(self) -> None:
        self.private: set[Path] = set()
        self.reparse: set[Path] = set()
        self.created: list[Path] = []

    def inspect(self, path: Path) -> PathInspection:
        if not path.exists():
            return PathInspection(PathKind.MISSING, False, None)
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            kind = PathKind.DIRECTORY
        elif stat.S_ISREG(metadata.st_mode):
            kind = PathKind.REGULAR_FILE
        else:
            kind = PathKind.OTHER
        return PathInspection(
            kind,
            path in self.reparse,
            (metadata.st_dev, metadata.st_ino),
        )

    def create_private_directory(self, path: Path) -> None:
        path.mkdir()
        self.created.append(path)
        self.private.add(path)

    def has_private_acl(self, path: Path) -> bool:
        return path in self.private

    def harden_acl(self, path: Path) -> None:
        self.private.add(path)


class _ChangingBackend(_Backend):
    """Report a substituted identity after the first target inspection."""

    def __init__(self, target: Path) -> None:
        super().__init__()
        self.target = target
        self.target_inspections = 0

    def inspect(self, path: Path) -> PathInspection:
        inspected = super().inspect(path)
        if path != self.target:
            return inspected
        self.target_inspections += 1
        if self.target_inspections == 1:
            return inspected
        assert inspected.identity is not None
        return PathInspection(
            inspected.kind,
            inspected.is_reparse_point,
            (inspected.identity[0], inspected.identity[1] + 1),
        )


def test_bounded_state_path_accepts_only_named_descendants(tmp_path: Path) -> None:
    assert bounded_state_path(tmp_path, "jobs/authority.v1.sqlite3") == (
        tmp_path / "jobs" / "authority.v1.sqlite3"
    )
    for unsafe in (
        "../escape.db",
        "jobs/../../escape.db",
        "authority.db:payload",
        "/absolute.db",
        "jobs/trailing. ",
        f"{'a' * 256}.db",
    ):
        with pytest.raises(StateSecurityError):
            bounded_state_path(tmp_path, unsafe)


def test_private_root_is_created_with_backend_acl_then_verified(
    tmp_path: Path,
) -> None:
    backend = _Backend()
    root = tmp_path / "authority"

    create_private_state_root(root, backend=backend)

    assert backend.created == [root]
    verify_state_root(root, backend=backend)


def test_existing_root_is_migrated_in_place_when_acl_is_not_private(
    tmp_path: Path,
) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    backend = _Backend()
    identity = backend.inspect(root).identity

    create_private_state_root(root, backend=backend)

    assert backend.inspect(root).identity == identity
    assert root in backend.private
    assert backend.created == []


def test_root_verification_rejects_reparse_in_every_path_component(
    tmp_path: Path,
) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    backend = _Backend()
    backend.private.add(root)
    backend.reparse.add(tmp_path)

    with pytest.raises(StateSecurityError) as captured:
        verify_state_root(root, backend=backend)

    assert captured.value.code is StateSecurityCode.REPARSE_POINT


def test_root_verification_rejects_identity_substitution(tmp_path: Path) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    backend = _ChangingBackend(root)
    backend.private.add(root)

    with pytest.raises(StateSecurityError) as captured:
        verify_state_root(root, backend=backend)

    assert captured.value.code is StateSecurityCode.IDENTITY_CHANGED


def test_file_verification_checks_root_bounds_type_acl_and_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    state_file = root / "authority.v1.sqlite3"
    state_file.write_bytes(b"state")
    backend = _Backend()
    backend.private.update((root, state_file))
    directory = root / "not-a-file"
    directory.mkdir()

    verify_state_file(root, state_file, backend=backend)

    with pytest.raises(StateSecurityError) as outside:
        verify_state_file(root, tmp_path / "outside.db", backend=backend)
    assert outside.value.code is StateSecurityCode.OUTSIDE_ROOT
    with pytest.raises(StateSecurityError) as wrong_type:
        verify_state_file(root, directory, backend=backend)
    assert wrong_type.value.code is StateSecurityCode.WRONG_TYPE


def test_file_verification_rejects_reparse_in_nested_component(
    tmp_path: Path,
) -> None:
    root = tmp_path / "authority"
    nested = root / "jobs"
    nested.mkdir(parents=True)
    state_file = nested / "state.db"
    state_file.write_bytes(b"state")
    backend = _Backend()
    backend.private.update((root, state_file))
    backend.reparse.add(nested)

    with pytest.raises(StateSecurityError) as captured:
        verify_state_file(root, state_file, backend=backend)

    assert captured.value.code is StateSecurityCode.REPARSE_POINT


@pytest.mark.skipif(os.name != "nt", reason="requires Windows security APIs")
def test_windows_backend_requires_explicit_protection_on_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "authority"

    create_private_state_root(root)
    state_file = root / "authority.v1.sqlite3"
    state_file.write_bytes(b"state")

    verify_state_root(root)
    with pytest.raises(StateSecurityError) as inherited:
        verify_state_file(root, state_file)
    assert inherited.value.code is StateSecurityCode.DACL_NOT_PRIVATE
    from rate_of_closure.web_authority._windows_state_security import (
        WindowsStateSecurityBackend,
    )

    WindowsStateSecurityBackend().harden_acl(state_file)
    verify_state_file(root, state_file)


@pytest.mark.skipif(os.name != "nt", reason="requires Windows security APIs")
def test_windows_existing_broad_root_is_hardened_without_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    state_file = root / "authority.v1.sqlite3"
    state_file.write_bytes(b"preserve-exactly")
    identity = state_file.stat().st_ino
    subprocess.run(
        ["icacls", str(root), "/grant", "*S-1-1-0:(OI)(CI)(RX)"],
        check=True,
        capture_output=True,
    )

    lease = prepare_private_state_root(root)
    lease.secure_files((state_file,))

    assert state_file.read_bytes() == b"preserve-exactly"
    assert state_file.stat().st_ino == identity
    verify_state_root(root)
    verify_state_file(root, state_file)
    lease.close()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows security APIs")
def test_windows_guard_blocks_secured_file_substitution(tmp_path: Path) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    state_file = root / "authority.v1.sqlite3"
    state_file.write_bytes(b"state")
    replacement = root / "replacement.sqlite3"
    replacement.write_bytes(b"replacement")
    lease = prepare_private_state_root(root)
    lease.secure_files((state_file,))

    with pytest.raises(PermissionError):
        os.replace(replacement, state_file)

    assert state_file.read_bytes() == b"state"
    lease.close()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows security APIs")
def test_windows_rejects_hard_linked_state_file(tmp_path: Path) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    state_file = root / "authority.v1.sqlite3"
    state_file.write_bytes(b"state")
    os.link(state_file, root / "second-name.sqlite3")
    lease = prepare_private_state_root(root)

    with pytest.raises(StateSecurityError) as captured:
        lease.secure_files((state_file,))

    assert captured.value.code is StateSecurityCode.HARD_LINK
    lease.close()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows security APIs")
def test_windows_rejects_planted_alternate_data_stream(tmp_path: Path) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    state_file = root / "authority.v1.sqlite3"
    state_file.write_bytes(b"state")
    Path(f"{state_file}:planted").write_bytes(b"hidden")
    lease = prepare_private_state_root(root)

    with pytest.raises(StateSecurityError) as captured:
        lease.secure_files((state_file,))

    assert captured.value.code is StateSecurityCode.UNEXPECTED_STREAM
    assert Path(f"{state_file}:planted").read_bytes() == b"hidden"
    lease.close()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows named streams")
def test_windows_rejects_planted_root_alternate_data_stream(tmp_path: Path) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    Path(f"{root}:planted").write_bytes(b"hidden")

    with pytest.raises(StateSecurityError) as captured:
        prepare_private_state_root(root)

    assert captured.value.code is StateSecurityCode.UNEXPECTED_STREAM
    assert Path(f"{root}:planted").read_bytes() == b"hidden"


@pytest.mark.skipif(os.name != "nt", reason="requires Windows error mapping")
def test_windows_maps_native_path_busy_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rate_of_closure.web_authority import _windows_state_native as native

    root = tmp_path / "authority"
    root.mkdir()
    state_file = root / "authority.v1.sqlite3"
    state_file.write_bytes(b"unchanged")
    lease = prepare_private_state_root(root)

    def busy(*_args: object, **_kwargs: object) -> object:
        raise pywintypes.error(32, "CreateFile", "injected")

    monkeypatch.setattr(native.win32file, "CreateFile", busy)
    with pytest.raises(StateSecurityError) as captured:
        lease.secure_files((state_file,))

    assert captured.value.code is StateSecurityCode.PATH_BUSY
    assert state_file.read_bytes() == b"unchanged"
    lease.close()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows error mapping")
def test_windows_maps_native_access_denied_without_path_disclosure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rate_of_closure.web_authority import _windows_state_native as native

    root = tmp_path / "authority"
    root.mkdir()
    state_file = root / "authority.v1.sqlite3"
    state_file.write_bytes(b"unchanged")
    lease = prepare_private_state_root(root)

    def deny(*_args: object, **_kwargs: object) -> object:
        raise pywintypes.error(5, "CreateFile", "injected")

    monkeypatch.setattr(native.win32file, "CreateFile", deny)
    with pytest.raises(StateSecurityError) as captured:
        lease.secure_files((state_file,))

    assert captured.value.code is StateSecurityCode.ACCESS_DENIED
    assert str(state_file) not in str(captured.value)
    assert state_file.read_bytes() == b"unchanged"
    lease.close()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows symlinks")
@pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
def test_windows_rejects_dangling_sidecar_symlink(
    tmp_path: Path,
    suffix: str,
) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    sidecar = root / f"authority.v1.sqlite3{suffix}"
    try:
        os.symlink(tmp_path / "missing-outside", sidecar)
    except OSError as error:
        if error.winerror != 1314:
            raise
        if os.environ.get("ROC_REQUIRE_WINDOWS_SYMLINK_TEST") == "1":
            pytest.fail("protected Windows lane lacks symlink privilege")
        pytest.skip("local account lacks Windows symlink privilege")
    lease = prepare_private_state_root(root)

    with pytest.raises(StateSecurityError) as captured:
        lease.secure_files((sidecar,))

    assert captured.value.code is StateSecurityCode.REPARSE_POINT
    lease.close()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows reparse points")
@pytest.mark.parametrize("kind", ["symlink", "junction"])
def test_windows_rejects_reparse_root_without_touching_target(
    tmp_path: Path,
    kind: str,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_bytes(b"unchanged")
    root = tmp_path / "authority"
    if kind == "symlink":
        try:
            os.symlink(target, root, target_is_directory=True)
        except OSError as error:
            if error.winerror != 1314:
                raise
            if os.environ.get("ROC_REQUIRE_WINDOWS_SYMLINK_TEST") == "1":
                pytest.fail("protected Windows lane lacks symlink privilege")
            pytest.skip("local account lacks Windows symlink privilege")
    else:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(root), str(target)],
            check=True,
            capture_output=True,
        )

    with pytest.raises(StateSecurityError) as captured:
        prepare_private_state_root(root)

    assert captured.value.code is StateSecurityCode.REPARSE_POINT
    assert sentinel.read_bytes() == b"unchanged"
    assert not (target / "authority.v1.sqlite3").exists()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows security descriptors")
@pytest.mark.parametrize("rollback_fails", [False, True])
def test_windows_acl_migration_rolls_back_prior_files_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rollback_fails: bool,
) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    first = root / "authority.v1.sqlite3"
    second = root / "authority.v1.sqlite3.lock"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    lease = prepare_private_state_root(root)
    original_descriptor = win32security.GetNamedSecurityInfo(
        str(first),
        win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION
        | win32security.DACL_SECURITY_INFORMATION,
    )
    original_sddl = win32security.ConvertSecurityDescriptorToStringSecurityDescriptor(
        original_descriptor,
        win32security.SDDL_REVISION_1,
        win32security.OWNER_SECURITY_INFORMATION
        | win32security.DACL_SECURITY_INFORMATION,
    )
    original_apply = lease._api.apply_private_acl
    calls = 0

    def fail_second(guard: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise StateSecurityError(StateSecurityCode.ACCESS_DENIED)
        original_apply(guard)

    monkeypatch.setattr(lease._api, "apply_private_acl", fail_second)
    if rollback_fails:
        monkeypatch.setattr(
            lease._api,
            "restore",
            lambda *_args: (_ for _ in ()).throw(
                StateSecurityError(StateSecurityCode.ACCESS_DENIED)
            ),
        )

    with pytest.raises(StateSecurityError) as captured:
        lease.secure_files((first, second))

    expected = (
        StateSecurityCode.ROLLBACK_INCOMPLETE
        if rollback_fails
        else StateSecurityCode.ACCESS_DENIED
    )
    assert captured.value.code is expected
    assert first.read_bytes() == b"first"
    assert second.read_bytes() == b"second"
    if not rollback_fails:
        from rate_of_closure.web_authority._windows_state_security import (
            WindowsStateSecurityBackend,
        )

        assert WindowsStateSecurityBackend().has_private_acl(first) is False
        restored_descriptor = win32security.GetNamedSecurityInfo(
            str(first),
            win32security.SE_FILE_OBJECT,
            win32security.OWNER_SECURITY_INFORMATION
            | win32security.DACL_SECURITY_INFORMATION,
        )
        restored_sddl = (
            win32security.ConvertSecurityDescriptorToStringSecurityDescriptor(
                restored_descriptor,
                win32security.SDDL_REVISION_1,
                win32security.OWNER_SECURITY_INFORMATION
                | win32security.DACL_SECURITY_INFORMATION,
            )
        )
        assert restored_sddl == original_sddl
    lease.close()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows retained handles")
def test_windows_later_hardening_failure_preserves_existing_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "authority"
    root.mkdir()
    database = root / "authority.v1.sqlite3"
    sidecar = root / "authority.v1.sqlite3-wal"
    database.write_bytes(b"database")
    lease = prepare_private_state_root(root)
    lease.secure_files((database,))
    original_guard = lease._files[database]
    sidecar.write_bytes(b"sidecar")

    monkeypatch.setattr(
        lease._api,
        "apply_private_acl",
        lambda _guard: (_ for _ in ()).throw(
            StateSecurityError(StateSecurityCode.ACCESS_DENIED)
        ),
    )
    with pytest.raises(StateSecurityError):
        lease.secure_files((database, sidecar))

    assert lease._files[database] is original_guard
    assert original_guard.handle is not None
    assert sidecar not in lease._files
    replacement = root / "replacement.sqlite3"
    replacement.write_bytes(b"replacement")
    with pytest.raises(PermissionError):
        os.replace(replacement, database)
    lease.close()
