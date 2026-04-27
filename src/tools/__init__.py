"""Tools package initialization."""

from __future__ import annotations

from typing import Any


def setup_logging(*args: Any, **kwargs: Any) -> Any:
    """Proxy to the shared logging helper without importing it at package load time."""
    from utils.logging_utils import setup_logging as _setup_logging

    return _setup_logging(*args, **kwargs)


__all__ = ["setup_logging"]
