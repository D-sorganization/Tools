"""Package: data_processor.

This outer package hosts launchers and metadata while the importable processing
package lives under ``python/data_processor``. Extend the package search path so
``import data_processor.core`` remains stable regardless of which source root is
first on ``sys.path``.
"""

from __future__ import annotations

from pathlib import Path

_nested_package = Path(__file__).resolve().parent / "python" / "data_processor"
if _nested_package.is_dir():
    __path__.append(str(_nested_package))
