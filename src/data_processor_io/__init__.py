"""Compatibility wrapper for ``shared.python.data_processor_io``.

The bulk-I/O wrapper historically exposed its own top-level package name while
delegating implementation to the shared package tree. Keep that public module
identity so it cannot shadow the full ``data_processor`` application package.
"""

from importlib import import_module

_CANONICAL_METADATA = {
    "__name__",
    "__package__",
    "__spec__",
    "__loader__",
    "__path__",
    "__file__",
    "__cached__",
}

_canonical = import_module("shared.python.data_processor_io")
__path__ = list(getattr(_canonical, "__path__", []))

for _name, _value in _canonical.__dict__.items():
    if _name not in _CANONICAL_METADATA:
        globals()[_name] = _value

del _name, _value, _canonical, import_module
