"""Scientific import detection uses Python syntax without executing source."""

from pathlib import Path

import pytest

from scripts.build_tools_module_inventory import _classification

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np",
        "import os, scipy.linalg as la",
        "from scipy.linalg import eig",
        "from numpy import (\n    array,\n    zeros,\n)",
        "import\tmath",
        "from statistics import mean",
        "if True:\n    import sympy",
        "try:\n    import casadi\nexcept ImportError:\n    pass",
        "raise RuntimeError('never execute inventory source')\nimport numpy",
    ],
)
def test_real_python_imports_are_provisional_calculation_signals(source: str) -> None:
    assert _classification(Path("src/widget.py"), source) == (
        "calculation",
        "scientific-library-import",
    )


@pytest.mark.parametrize(
    "source",
    [
        "# importnumpy\n# import numpy",
        "text = 'importnumpy'",
        '"""Example: importnumpy\nimport numpy\n"""',
        "importnumpy = 1",
        "import numpy_extra",
        "import NUMPY",
        "from local.scipy import value",
        "from .numpy import array",
        "from . import scipy",
        "import json",
    ],
)
def test_nonimports_and_unrelated_or_relative_names_do_not_make_signals(
    source: str,
) -> None:
    assert _classification(Path("src/widget.py"), source) == (
        "non-calculation",
        "no-conservative-calculation-signal",
    )


def test_unparseable_python_cannot_silently_claim_noncalculation() -> None:
    with pytest.raises(ValueError, match="Python.*import"):
        _classification(Path("src/widget.py"), "from numpy import (")


def test_python_suffix_case_matches_inventory_language_detection() -> None:
    assert _classification(Path("src/widget.PY"), "import numpy")[0] == "calculation"


def test_existing_path_and_nonpython_signals_are_retained() -> None:
    assert _classification(Path("src/dynamics.py"), "import json") == (
        "calculation",
        "path-marker:dynamics",
    )
    assert _classification(Path("src/widget.cpp"), "#include <math.h>") == (
        "calculation",
        "scientific-library-import",
    )
