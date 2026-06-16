"""Compatibility shim for the canonical ``shared.python.logging_pkg`` package."""

from shared.python.import_aliases import alias_legacy_package

_canonical = alias_legacy_package(__name__, "shared.python.logging_pkg")
globals().update(_canonical.__dict__)
