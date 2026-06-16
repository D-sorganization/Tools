"""Compatibility shim for the canonical ``shared.python.calc_backend`` package."""

from shared.python.import_aliases import alias_legacy_package

_canonical = alias_legacy_package(__name__, "shared.python.calc_backend")
globals().update(_canonical.__dict__)
