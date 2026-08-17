"""Platform-specific process-lifetime security for an authority job store."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from .state_security import PrivateStateRoot, prepare_private_state_root

_SIDE_SUFFIXES: Final = ("-wal", "-shm", "-journal")


class JobStoreSecurity:
    """Own hardening and retained handles around one SQLite store."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = Path(f"{path}.lock")
        self._lease: PrivateStateRoot | None = None
        if os.name == "nt":
            self._prepare_windows()
        else:
            path.parent.chmod(0o700)

    def _prepare_windows(self) -> None:
        lease = prepare_private_state_root(self.path.parent)
        self._lease = lease
        names = frozenset(candidate.name for candidate in self._paths())
        lease.assert_only_entries(names)
        try:
            self._create_if_missing(self.path)
            self._create_if_missing(self.lock_path)
            lease.secure_files((self.path, self.lock_path))
        except Exception:
            self.close()
            raise

    @staticmethod
    def _create_if_missing(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return
        os.close(descriptor)

    def _paths(self) -> tuple[Path, ...]:
        return (
            self.path,
            *(Path(f"{self.path}{suffix}") for suffix in _SIDE_SUFFIXES),
            self.lock_path,
        )

    def harden_files(self) -> None:
        """Secure each currently materialized database artifact."""
        if self._lease is not None:
            self._lease.secure_files(self._paths())
            return
        for candidate in self._paths():
            if candidate.exists():
                candidate.chmod(0o600)

    def before_connection_close(self) -> None:
        """Let SQLite remove transient sidecars during orderly shutdown."""
        if self._lease is not None:
            self._lease.release_transient_files()

    def close(self) -> None:
        lease, self._lease = self._lease, None
        if lease is not None:
            lease.close()


__all__ = ["JobStoreSecurity"]
