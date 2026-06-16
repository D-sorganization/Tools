"""Compatibility shim for the canonical ``shared.python.data_processor_io`` package."""

from shared.python.import_aliases import alias_legacy_package

_canonical = alias_legacy_package(__name__, "shared.python.data_processor_io")
globals().update(_canonical.__dict__)
