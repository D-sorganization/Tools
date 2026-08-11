"""Private type aliases shared by plotting catalog data modules."""

from rate_of_closure.plotting.catalog_contract import Extractor

CatalogRow = tuple[str, str, str, Extractor]
CatalogGroup = tuple[str, str, tuple[CatalogRow, ...]]
