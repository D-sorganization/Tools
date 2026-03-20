"""Sphinx configuration for Double Pendulum Golf Simulator."""

import sys
from os.path import abspath

# Add source directory to path for autodoc
_src_path = abspath("../src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# -- Project information -------------------------------------------------------
project = "Double Pendulum Golf Simulator"
copyright = "2026, Dieter Olson"
author = "Dieter Olson"
release = "0.1.0"

# -- General configuration -----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # "sphinx.ext.intersphinx",  # enable when publishing online
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Napoleon settings (Google/NumPy docstrings)
napoleon_google_docstrings = True
napoleon_numpy_docstrings = True
napoleon_include_init_with_doc = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

# Mock imports for headless builds (no PyQt6)
autodoc_mock_imports = [
    "PyQt6",
    "PyQt6.QtWidgets",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtSvg",
    "PyQt6.QtPrintSupport",
    "pyqtgraph",
    "jax",
    "jaxlib",
    "diffrax",
    "optax",
]

# Intersphinx (uncomment when publishing online)
# intersphinx_mapping = {
#     "python": ("https://docs.python.org/3", None),
#     "numpy": ("https://numpy.org/doc/stable/", None),
#     "scipy": ("https://docs.scipy.org/doc/scipy/", None),
# }

# -- Options for HTML output ---------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
