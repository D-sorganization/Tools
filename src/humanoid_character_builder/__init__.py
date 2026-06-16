"""Compatibility shim for ``shared.python.humanoid_character_builder``."""

from shared.python.import_aliases import alias_legacy_package

_canonical = alias_legacy_package(__name__, "shared.python.humanoid_character_builder")
globals().update(_canonical.__dict__)
