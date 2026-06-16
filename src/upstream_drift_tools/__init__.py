"""Deprecated compatibility shim for ``shared.python.sidekick``."""

from shared.python.import_aliases import alias_legacy_package

_canonical = alias_legacy_package(
    __name__,
    "shared.python.sidekick",
    warning=(
        "upstream_drift_tools is deprecated and will be removed in a future "
        "release. Import from shared.python.sidekick instead."
    ),
)
globals().update(_canonical.__dict__)
