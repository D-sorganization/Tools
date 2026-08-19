"""Fail-closed filesystem security primitives for authority state."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol, cast

MAX_STATE_PATH_CHARS: Final = 1_024
MAX_STATE_COMPONENT_CHARS: Final = 255
_RESERVED_NAMES: Final = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


class StateSecurityCode(StrEnum):
    """Stable, non-sensitive categories for authority-state failures."""

    INVALID_PATH = "invalid_path"
    ALTERNATE_DATA_STREAM = "alternate_data_stream"
    OUTSIDE_ROOT = "outside_root"
    MISSING = "missing"
    WRONG_TYPE = "wrong_type"
    REPARSE_POINT = "reparse_point"
    HARD_LINK = "hard_link"
    UNEXPECTED_STREAM = "unexpected_stream"
    ACCESS_DENIED = "access_denied"
    PATH_BUSY = "path_busy"
    UNSUPPORTED_VOLUME = "unsupported_volume"
    UNEXPECTED_ENTRY = "unexpected_entry"
    ROLLBACK_INCOMPLETE = "rollback_incomplete"
    DACL_NOT_PRIVATE = "dacl_not_private"
    IDENTITY_CHANGED = "identity_changed"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    OPERATING_SYSTEM_FAILURE = "operating_system_failure"


_FAILURE_MESSAGES: Final = {
    StateSecurityCode.INVALID_PATH: "authority state path is invalid",
    StateSecurityCode.ALTERNATE_DATA_STREAM: (
        "authority state path contains an alternate data stream"
    ),
    StateSecurityCode.OUTSIDE_ROOT: "authority state path escapes its root",
    StateSecurityCode.MISSING: "authority state path is missing",
    StateSecurityCode.WRONG_TYPE: "authority state path has an unsafe type",
    StateSecurityCode.REPARSE_POINT: "authority state path contains a reparse point",
    StateSecurityCode.HARD_LINK: "authority state file has multiple names",
    StateSecurityCode.UNEXPECTED_STREAM: (
        "authority state contains an unexpected stream"
    ),
    StateSecurityCode.ACCESS_DENIED: "authority state access was denied",
    StateSecurityCode.PATH_BUSY: "authority state path is already in use",
    StateSecurityCode.UNSUPPORTED_VOLUME: "authority state volume is unsupported",
    StateSecurityCode.UNEXPECTED_ENTRY: (
        "authority state directory contains an unexpected entry"
    ),
    StateSecurityCode.ROLLBACK_INCOMPLETE: (
        "authority state permission rollback was incomplete"
    ),
    StateSecurityCode.DACL_NOT_PRIVATE: "authority state permissions are not private",
    StateSecurityCode.IDENTITY_CHANGED: "authority state path identity changed",
    StateSecurityCode.UNSUPPORTED_PLATFORM: (
        "authority state security is unsupported on this platform"
    ),
    StateSecurityCode.OPERATING_SYSTEM_FAILURE: (
        "authority state security operation failed"
    ),
}


class StateSecurityError(RuntimeError):
    """Typed failure that never includes the sensitive filesystem path."""

    def __init__(self, code: StateSecurityCode) -> None:
        if type(code) is not StateSecurityCode:
            raise TypeError("state security code must be exact")
        self.code = code
        super().__init__(_FAILURE_MESSAGES[code])


class PathKind(StrEnum):
    """Filesystem kinds relevant to state authority validation."""

    MISSING = "missing"
    DIRECTORY = "directory"
    REGULAR_FILE = "regular_file"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class PathInspection:
    """One path's type, redirection state, and stable filesystem identity."""

    kind: PathKind
    is_reparse_point: bool
    identity: tuple[int, int] | None

    def __post_init__(self) -> None:
        if type(self.kind) is not PathKind:
            raise TypeError("path inspection kind must be exact")
        has_identity = self.identity is not None
        if has_identity == (self.kind is PathKind.MISSING):
            raise ValueError("existing paths require an identity")


class StateSecurityBackend(Protocol):
    """Minimal backend boundary for filesystem and native ACL operations."""

    def inspect(self, path: Path) -> PathInspection:
        """Return non-following metadata for one absolute path."""

    def create_private_directory(self, path: Path) -> None:
        """Atomically create one directory with a private DACL."""

    def has_private_acl(self, path: Path) -> bool:
        """Return whether only owner, Administrators, and SYSTEM have access."""

    def harden_acl(self, path: Path) -> None:
        """Replace the path DACL with the exact private contract in place."""


class PrivateStateRoot(Protocol):
    """Process-lifetime guard for one private Windows authority state root."""

    def secure_files(self, paths: tuple[Path, ...]) -> None:
        """Harden existing files and retain anti-substitution handles."""

    def assert_only_entries(self, names: frozenset[str]) -> None:
        """Fail if the root contains a name outside the fixed allowlist."""

    def release_transient_files(self) -> None:
        """Release SQLite sidecar guards before connection shutdown."""

    def close(self) -> None:
        """Release every retained native handle."""


def _fail(code: StateSecurityCode) -> None:
    raise StateSecurityError(code)


def _validate_component(component: str) -> None:
    if ":" in component:
        _fail(StateSecurityCode.ALTERNATE_DATA_STREAM)
    stem = component.split(".", maxsplit=1)[0].upper()
    invalid = (
        component in {"", ".", ".."}
        or len(component) > MAX_STATE_COMPONENT_CHARS
        or component.endswith((" ", "."))
        or stem in _RESERVED_NAMES
    )
    if invalid:
        _fail(StateSecurityCode.INVALID_PATH)


def _validate_absolute_path(path: Path) -> None:
    if not isinstance(path, Path) or not path.is_absolute() or path.name == "":
        _fail(StateSecurityCode.INVALID_PATH)
    source = str(path)
    if len(source) > MAX_STATE_PATH_CHARS or path.anchor.startswith("\\\\"):
        _fail(StateSecurityCode.INVALID_PATH)
    for component in path.parts[1:]:
        _validate_component(component)


def bounded_state_path(root: Path, relative: str) -> Path:
    """Build one lexical descendant after enforcing portable Windows bounds.

    Preconditions:
        ``root`` is a named local absolute path and ``relative`` is text.
    Postconditions:
        The result is a named descendant of ``root`` without ADS syntax.
    """
    _validate_absolute_path(root)
    if type(relative) is not str or not relative:
        _fail(StateSecurityCode.INVALID_PATH)
    relative_path = Path(relative)
    if relative_path.is_absolute() or relative_path.anchor:
        _fail(StateSecurityCode.INVALID_PATH)
    for component in relative_path.parts:
        _validate_component(component)
    candidate = root.joinpath(relative_path)
    _validate_absolute_path(candidate)
    if not candidate.is_relative_to(root) or candidate == root:
        _fail(StateSecurityCode.OUTSIDE_ROOT)
    return candidate


def _default_backend() -> StateSecurityBackend:
    if os.name != "nt":
        _fail(StateSecurityCode.UNSUPPORTED_PLATFORM)
    # Cast because the import is deliberately deferred: `_windows_state_security`
    # is Windows-only, so it cannot be imported at module scope, and the
    # changed-file MyPy gate's `--follow-imports=skip` resolves the deferred
    # module to `Any`. The class does implement the protocol.
    from ._windows_state_security import WindowsStateSecurityBackend

    return cast("StateSecurityBackend", WindowsStateSecurityBackend())


def _select_backend(
    backend: StateSecurityBackend | None,
) -> StateSecurityBackend:
    return backend if backend is not None else _default_backend()


def _path_chain(path: Path) -> tuple[Path, ...]:
    anchor = Path(path.anchor)
    chain = [anchor]
    for component in path.parts[1:]:
        chain.append(chain[-1] / component)
    return tuple(chain)


def _inspect_directory_chain(
    path: Path,
    backend: StateSecurityBackend,
) -> tuple[PathInspection, ...]:
    inspections = tuple(backend.inspect(candidate) for candidate in _path_chain(path))
    for inspected in inspections:
        if inspected.kind is PathKind.MISSING:
            _fail(StateSecurityCode.MISSING)
        if inspected.kind is not PathKind.DIRECTORY:
            _fail(StateSecurityCode.WRONG_TYPE)
        if inspected.is_reparse_point:
            _fail(StateSecurityCode.REPARSE_POINT)
    return inspections


def _assert_unchanged(
    before: tuple[PathInspection, ...],
    after: tuple[PathInspection, ...],
) -> None:
    if tuple(item.identity for item in before) != tuple(
        item.identity for item in after
    ):
        _fail(StateSecurityCode.IDENTITY_CHANGED)


def _inspect_file_chain(
    path: Path,
    backend: StateSecurityBackend,
) -> tuple[PathInspection, ...]:
    inspections = tuple(backend.inspect(candidate) for candidate in _path_chain(path))
    for inspected in inspections[:-1]:
        if inspected.kind is not PathKind.DIRECTORY:
            _fail(StateSecurityCode.WRONG_TYPE)
        if inspected.is_reparse_point:
            _fail(StateSecurityCode.REPARSE_POINT)
    target = inspections[-1]
    if target.kind is PathKind.MISSING:
        _fail(StateSecurityCode.MISSING)
    if target.kind is not PathKind.REGULAR_FILE:
        _fail(StateSecurityCode.WRONG_TYPE)
    if target.is_reparse_point:
        _fail(StateSecurityCode.REPARSE_POINT)
    return inspections


def verify_state_root(
    root: Path,
    *,
    backend: StateSecurityBackend | None = None,
) -> None:
    """Verify a private directory and every component leading to it.

    Preconditions:
        ``root`` is a named local absolute path.
    Postconditions:
        The root is an identity-stable, non-reparse directory with a private ACL.
    """
    _validate_absolute_path(root)
    selected = _select_backend(backend)
    before = _inspect_directory_chain(root, selected)
    if not selected.has_private_acl(root):
        _fail(StateSecurityCode.DACL_NOT_PRIVATE)
    after = _inspect_directory_chain(root, selected)
    _assert_unchanged(before, after)


def create_private_state_root(
    root: Path,
    *,
    backend: StateSecurityBackend | None = None,
) -> None:
    """Create one private state root, or fail closed on an existing unsafe root.

    Preconditions:
        The root parent already exists and contains no reparse component.
    Postconditions:
        ``verify_state_root`` succeeds for the created or existing root.
    """
    _validate_absolute_path(root)
    selected = _select_backend(backend)
    _inspect_directory_chain(root.parent, selected)
    inspected = selected.inspect(root)
    if inspected.kind is PathKind.MISSING:
        selected.create_private_directory(root)
    elif inspected.kind is not PathKind.DIRECTORY:
        _fail(StateSecurityCode.WRONG_TYPE)
    elif inspected.is_reparse_point:
        _fail(StateSecurityCode.REPARSE_POINT)
    elif not selected.has_private_acl(root):
        selected.harden_acl(root)
    verify_state_root(root, backend=selected)


def prepare_private_state_root(root: Path) -> PrivateStateRoot:
    """Create, migrate, verify, and retain a Windows state-root lease."""
    _validate_absolute_path(root)
    if os.name != "nt":
        _fail(StateSecurityCode.UNSUPPORTED_PLATFORM)
    # Cast for the same reason as `_default_backend`: the Windows-only module is
    # imported lazily and resolves to `Any` under `--follow-imports=skip`.
    from ._windows_state_security import WindowsPrivateStateRoot

    return cast("PrivateStateRoot", WindowsPrivateStateRoot(root))


def verify_state_file(
    root: Path,
    path: Path,
    *,
    backend: StateSecurityBackend | None = None,
) -> None:
    """Verify one private regular file bounded below a verified state root.

    Preconditions:
        ``root`` and ``path`` are named local absolute paths.
    Postconditions:
        The root and file chain are identity-stable, non-reparse, and private.
    """
    _validate_absolute_path(root)
    _validate_absolute_path(path)
    if path == root or not path.is_relative_to(root):
        _fail(StateSecurityCode.OUTSIDE_ROOT)
    selected = _select_backend(backend)
    verify_state_root(root, backend=selected)
    before = _inspect_file_chain(path, selected)
    if not selected.has_private_acl(path):
        _fail(StateSecurityCode.DACL_NOT_PRIVATE)
    after = _inspect_file_chain(path, selected)
    _assert_unchanged(before, after)
