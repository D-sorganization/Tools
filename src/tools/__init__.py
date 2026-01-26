"""Tools package initialization."""

from tools.logger import setup_logging

# Expose setup_logging at package level
__all__ = ["setup_logging"]
