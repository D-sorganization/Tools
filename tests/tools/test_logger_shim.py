"""Tests for the deprecated tools/logger.py shim module."""

import warnings


def test_logger_shim_issues_deprecation_warning():
    """Importing tools.logger must emit a DeprecationWarning."""
    import sys

    # Remove any cached import so the warning fires fresh
    for mod in list(sys.modules.keys()):
        if "tools.logger" in mod or mod == "tools.logger":
            del sys.modules[mod]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import tools.logger  # noqa: F401

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any(
            "tools.logger" in str(w.message) for w in dep_warnings
        ), "Expected DeprecationWarning about tools.logger"


def test_logger_shim_re_exports_setup_logging():
    """tools.logger must re-export setup_logging."""
    import sys

    for mod in list(sys.modules.keys()):
        if "tools.logger" in mod or mod == "tools.logger":
            del sys.modules[mod]

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        import tools.logger as shim

    assert hasattr(shim, "setup_logging")
    assert callable(shim.setup_logging)


def test_logger_shim_re_exports_get_logger():
    """tools.logger must re-export get_logger."""
    import sys

    for mod in list(sys.modules.keys()):
        if "tools.logger" in mod or mod == "tools.logger":
            del sys.modules[mod]

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        import tools.logger as shim

    assert hasattr(shim, "get_logger")
    assert callable(shim.get_logger)


def test_logger_shim_all_exports():
    """tools.logger.__all__ must contain the expected symbols."""
    import sys

    for mod in list(sys.modules.keys()):
        if "tools.logger" in mod or mod == "tools.logger":
            del sys.modules[mod]

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        import tools.logger as shim

    assert "DEFAULT_FORMAT" in shim.__all__
    assert "get_logger" in shim.__all__
    assert "setup_logging" in shim.__all__
