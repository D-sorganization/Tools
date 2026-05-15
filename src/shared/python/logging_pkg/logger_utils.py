"""Logger utilities shim for backward compatibility.

Re-exports from logging_config to satisfy existing imports.
"""

from .logging_config import get_logger

__all__ = ["get_logger"]