"""Source-backed, subscription-free developer context owned by Tools.

Metadata and indexes aid discovery; source code, tests and reviewed scientific
documentation retain their authority. Importing this package has no side effects.
"""

from .catalog import CatalogError, load_catalog

__all__ = ["CatalogError", "load_catalog"]
