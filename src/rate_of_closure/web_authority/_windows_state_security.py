"""Windows leases for private authority-state paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from ._windows_state_native import NativeApi, NativeGuard, SecuritySnapshot
from .state_security import (
    PathInspection,
    PathKind,
    StateSecurityCode,
    StateSecurityError,
)

_TRANSIENT_SUFFIXES: Final = ("-wal", "-shm", "-journal")


def _fail(code: StateSecurityCode) -> None:
    raise StateSecurityError(code)


def _chain(path: Path) -> tuple[Path, ...]:
    current = Path(path.anchor)
    values = [current]
    for component in path.parts[1:]:
        current /= component
        values.append(current)
    return tuple(values)


class WindowsPrivateStateRoot:
    """Harden a root and retain handles that block pathname substitution."""

    def __init__(self, root: Path) -> None:
        if os.name != "nt":
            _fail(StateSecurityCode.UNSUPPORTED_PLATFORM)
        self.root = root
        self._api = NativeApi()
        self._ancestors: list[NativeGuard] = []
        self._files: dict[Path, NativeGuard] = {}
        try:
            self._open_or_create_chain()
            self._api.verify_volume(self._ancestors[0])
            self._api.assert_default_stream_only(self._ancestors[-1])
            self._harden_batch((self._ancestors[-1],))
        except Exception:
            self.close()
            raise

    def _open_or_create_chain(self) -> None:
        for path in _chain(self.root):
            try:
                guard = self._api.open_guard(path, security=path == self.root)
            except StateSecurityError as error:
                if error.code is not StateSecurityCode.MISSING:
                    raise
                if path != self.root:
                    raise
                self._api.create_private_directory(path)
                guard = self._api.open_guard(path, security=True)
            if not guard.is_directory:
                guard.close()
                _fail(StateSecurityCode.WRONG_TYPE)
            self._ancestors.append(guard)

    def secure_files(self, paths: tuple[Path, ...]) -> None:
        """Harden existing regular files and retain no-delete handles."""
        candidates: list[NativeGuard] = []
        opened: list[Path] = []
        try:
            for path in paths:
                self._validate_descendant(path)
                if path in self._files:
                    candidates.append(self._files[path])
                    continue
                try:
                    guard = self._api.open_guard(path, security=True)
                except StateSecurityError as error:
                    if error.code is StateSecurityCode.MISSING:
                        continue
                    raise
                try:
                    if guard.is_directory:
                        _fail(StateSecurityCode.WRONG_TYPE)
                    if guard.link_count != 1:
                        _fail(StateSecurityCode.HARD_LINK)
                    self._api.assert_default_stream_only(guard)
                except Exception:
                    guard.close()
                    raise
                self._files[path] = guard
                opened.append(path)
                candidates.append(guard)
            self._harden_batch(tuple(candidates))
        except Exception:
            self._close_new_files(tuple(opened))
            raise

    def _validate_descendant(self, path: Path) -> None:
        if not isinstance(path, Path) or path.parent != self.root or path == self.root:
            _fail(StateSecurityCode.OUTSIDE_ROOT)

    def _harden_batch(self, guards: tuple[NativeGuard, ...]) -> None:
        snapshots = [(guard, self._api.snapshot(guard.handle)) for guard in guards]
        changed: list[tuple[NativeGuard, SecuritySnapshot]] = []
        try:
            for guard, snapshot in snapshots:
                if not self._api.has_private_acl(guard):
                    self._api.apply_private_acl(guard)
                    changed.append((guard, snapshot))
                if not self._api.has_private_acl(guard):
                    _fail(StateSecurityCode.DACL_NOT_PRIVATE)
        except Exception:
            self._rollback(tuple(reversed(changed)))
            raise

    def _rollback(
        self,
        changed: tuple[tuple[NativeGuard, SecuritySnapshot], ...],
    ) -> None:
        incomplete = False
        for guard, snapshot in changed:
            try:
                self._api.restore(guard, snapshot)
            except StateSecurityError:
                incomplete = True
        if incomplete:
            _fail(StateSecurityCode.ROLLBACK_INCOMPLETE)

    def _close_new_files(self, paths: tuple[Path, ...]) -> None:
        for path in paths:
            guard = self._files.pop(path, None)
            if guard is not None:
                guard.close()

    def assert_only_entries(self, names: frozenset[str]) -> None:
        """Reject unexpected names in a dedicated authority root."""
        actual = {entry.name for entry in self.root.iterdir()}
        if not actual <= names:
            _fail(StateSecurityCode.UNEXPECTED_ENTRY)

    def release_transient_files(self) -> None:
        """Release WAL/SHM/journal handles so SQLite may remove them."""
        for path, guard in tuple(self._files.items()):
            if path.name.endswith(_TRANSIENT_SUFFIXES):
                guard.close()
                del self._files[path]

    def close(self) -> None:
        for guard in tuple(self._files.values()):
            guard.close()
        self._files.clear()
        for guard in reversed(self._ancestors):
            guard.close()
        self._ancestors.clear()


class WindowsStateSecurityBackend:
    """Compatibility façade for focused verification helpers and tests."""

    def __init__(self) -> None:
        if os.name != "nt":
            _fail(StateSecurityCode.UNSUPPORTED_PLATFORM)
        self._api = NativeApi()

    def inspect(self, path: Path) -> PathInspection:
        if not path.exists() and not path.is_symlink():
            return PathInspection(PathKind.MISSING, False, None)
        guard = self._api.open_guard(path, security=False)
        try:
            kind = PathKind.DIRECTORY if guard.is_directory else PathKind.REGULAR_FILE
            return PathInspection(
                kind,
                False,
                (guard.identity.volume_serial, guard.identity.file_index),
            )
        finally:
            guard.close()

    def create_private_directory(self, path: Path) -> None:
        self._api.create_private_directory(path)

    def has_private_acl(self, path: Path) -> bool:
        guard = self._api.open_guard(path, security=True)
        try:
            return self._api.has_private_acl(guard)
        finally:
            guard.close()

    def harden_acl(self, path: Path) -> None:
        guard = self._api.open_guard(path, security=True)
        snapshot = self._api.snapshot(guard.handle)
        try:
            self._api.apply_private_acl(guard)
            if not self._api.has_private_acl(guard):
                _fail(StateSecurityCode.DACL_NOT_PRIVATE)
        except Exception:
            self._api.restore(guard, snapshot)
            raise
        finally:
            guard.close()
