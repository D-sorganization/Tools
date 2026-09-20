"""PyInstaller hook for rate_of_closure.

Ensures all assets, schemas, manifests, fixtures, and submodules are bundled.
"""

from __future__ import annotations

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files("rate_of_closure", include_py_files=False)
hiddenimports = collect_submodules("rate_of_closure")
