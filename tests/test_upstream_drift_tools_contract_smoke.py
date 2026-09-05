"""Minimum test-contract coverage for upstream_drift_tools package."""

from importlib import import_module


def test_upstream_drift_tools_contract_smoke() -> None:
    """Verify the upstream_drift_tools package can be imported as a package."""
    module = import_module("upstream_drift_tools")
    assert module.__name__ in ("upstream_drift_tools", "shared.python.sidekick")
    assert hasattr(module, "__path__")
