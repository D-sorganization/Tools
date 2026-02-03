"""FastAPI-based REST API for the Data Processor.

This API enables React and other frontends to access data processing
capabilities via HTTP endpoints.
"""

from .app import create_app

__all__ = ["create_app"]
