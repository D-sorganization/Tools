"""Compatibility shim for the canonical ``shared.python.file_watcher`` package."""

from shared.python.import_aliases import alias_legacy_package

_canonical = alias_legacy_package(__name__, "shared.python.file_watcher")
globals().update(_canonical.__dict__)
