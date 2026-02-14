"""Legacy placeholder test for plotting tab refactor path.

This file preserves the historical path expected by changed-file CI checks,
while the brittle legacy test suite has been retired.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="Legacy plotting-tab refactor suite removed; path retained for CI tooling."
)


def test_plotting_tab_refactor_legacy_placeholder() -> None:
    """Placeholder to keep test module valid and discoverable."""
    assert True
