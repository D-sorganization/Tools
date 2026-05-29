"""Unit tests for the ``data_processing`` package ``__init__`` lazy exports.

The package eagerly exposes its exception hierarchy and lazily resolves the
heavier engine/IO classes through ``__getattr__`` to avoid importing pandas at
package-import time. Tests verify both paths plus the unknown-attribute guard.
"""

from __future__ import annotations

import pytest
import sidekick.data_processing as dp


def test_exceptions_are_eagerly_available() -> None:
    assert issubclass(dp.ColumnNotFoundError, dp.DataProcessingError)
    assert issubclass(dp.DataNotLoadedError, dp.DataProcessingError)
    assert issubclass(dp.FileIOError, dp.DataProcessingError)


@pytest.mark.parametrize(
    "name",
    [
        "DataProcessorEngine",
        "ProcessingResult",
        "DataReader",
        "DataWriter",
        "FileFormatDetector",
    ],
)
def test_lazy_exports_resolve(name: str) -> None:
    obj = getattr(dp, name)
    assert obj is not None
    # Second access hits the cached global, exercising the fast path too.
    assert getattr(dp, name) is obj


def test_unknown_attribute_raises() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = dp.DoesNotExist


def test_all_is_complete() -> None:
    assert "DataProcessorEngine" in dp.__all__
    assert "ColumnNotFoundError" in dp.__all__
