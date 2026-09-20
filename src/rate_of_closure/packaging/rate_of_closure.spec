# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller specification for Rate of Closure Impact Explorer (onedir layout)."""

from pathlib import Path
from PyInstaller.utils.hooks import collect_all

_spec_dir = Path(SPECPATH).resolve()
_repo_root = _spec_dir.parents[2]
_src_dir = _repo_root / "src"

datas = []
binaries = []
hiddenimports = [
    "rate_of_closure",
    "rate_of_closure.packaging.entrypoint",
    "rate_of_closure.ui.pyqt6.main_window",
    "shared.python.swing_sim",
    "shared.python.contracts",
    "shared.python.theme",
    "shared.python.gui_launcher",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.QtSvg",
    "PyQt6.QtSvgWidgets",
    "matplotlib",
    "matplotlib.backends.backend_qtagg",
    "scipy",
    "scipy.spatial.transform",
    "scipy.optimize",
    "scipy.interpolate",
    "numpy",
    "pandas",
    "fastapi",
    "uvicorn",
]

_roc_collected = collect_all("rate_of_closure")
datas += _roc_collected[0]
binaries += _roc_collected[1]
hiddenimports += _roc_collected[2]

# Ensure model fixtures are included for qualification parity assertions
_fixtures_src = _src_dir / "rate_of_closure" / "web" / "src" / "model" / "__fixtures__"
if _fixtures_src.is_dir():
    datas.append((str(_fixtures_src), "rate_of_closure/web/src/model/__fixtures__"))

a = Analysis(
    [str(_spec_dir / "entrypoint.py")],
    pathex=[str(_src_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(_spec_dir / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PySide6",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtWidgets",
        "shiboken6",
        "PySide2",
        "PyQt5",
        "tkinter",
        "IPython",
        "notebook",
        "torch",
        "vtk",
        "vtkmodules",
        "pyvista",
        "pyvistaqt",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RateOfClosureExplorer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="RateOfClosureExplorer",
)
