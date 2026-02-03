"""API route modules."""

from .export import register_export_on_processing
from .export import router as export_router
from .files import router as files_router
from .processing import router as processing_router

# Register export endpoint on processing router for convenience
register_export_on_processing(processing_router)

__all__ = ["files_router", "processing_router", "export_router"]
