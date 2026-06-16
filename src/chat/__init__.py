"""Compatibility shim for the canonical ``shared.python.chat`` package."""

from shared.python.import_aliases import alias_legacy_package

_canonical = alias_legacy_package(__name__, "shared.python.chat")
globals().update(_canonical.__dict__)
