"""Rust-backed electrode advisor interface.

This module provides a clean Python facade over the ``tools_core.electrode_advisor``
Rust module (built via PyO3/Maturin).
"""

import logging
import os
import warnings
from typing import Any

logger = logging.getLogger(__name__)

_BACKEND_ENV = "GAS_THERMO_BACKEND"
_VALID_BACKENDS = ("rust", "python", "auto")
_DEFAULT_BACKEND = "auto"


def _resolve_backend() -> str:
    raw = os.environ.get(_BACKEND_ENV, _DEFAULT_BACKEND).strip().lower()
    if raw not in _VALID_BACKENDS:
        return _DEFAULT_BACKEND
    return raw


_REQUESTED_BACKEND = _resolve_backend()
_RUST_AVAILABLE = False
_rust: Any = None

if _REQUESTED_BACKEND != "python":
    try:
        from tools_core import electrode_advisor

        _rust = electrode_advisor

        _RUST_AVAILABLE = True
    except ImportError as e:
        if _REQUESTED_BACKEND == "rust":
            raise ImportError(
                "Rust tools_core wheel is required but not installed."
            ) from e


def is_rust_available() -> bool:
    return _RUST_AVAILABLE


def build_default_electrode_advisor_layout() -> Any:
    if _RUST_AVAILABLE:
        return _rust.build_default_electrode_advisor_layout()

    from .electrode_advisor import _py_build_default_electrode_advisor_layout

    warnings.warn(
        "Using slow pure-Python electrode advisor backend",
        DeprecationWarning,
        stacklevel=2,
    )
    return _py_build_default_electrode_advisor_layout()
