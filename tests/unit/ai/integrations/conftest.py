"""Pytest configuration for AI integration tests.

Provides VCR (Video Cassette Recorder) configuration so the Linear,
Notion, and Affine integration tests can replay HTTP traffic from
pre-recorded YAML cassettes without requiring live API credentials.

Notes
-----
- ``record_mode="none"`` ensures CI never attempts to record new
  cassettes against a live API; tests fail loudly if a request does
  not match the cassette.
- ``filter_headers`` strips any authorization material that may have
  leaked into cassettes during local recording.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: follow the pattern used in tests/unit/ai/integrations/
# test_obsidian_vault.py so ``src.shared.python.*`` imports resolve
# without requiring real ``__init__.py`` files on every ancestor.
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PACKAGE_STUBS: list[tuple[str, str | None]] = [
    ("src", "src"),
    ("src.shared", "src/shared"),
    ("src.shared.python", "src/shared/python"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.integrations", "src/shared/python/ai/integrations"),
]
for _mod_name, _rel_path in _PACKAGE_STUBS:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        if _rel_path is not None:
            _stub.__path__ = [str(_ROOT / _rel_path)]
        sys.modules[_mod_name] = _stub

sys.modules.setdefault(
    "src.shared.python.logging_pkg",
    types.ModuleType("src.shared.python.logging_pkg"),
)
_logging_config_stub = sys.modules.setdefault(
    "src.shared.python.logging_pkg.logging_config",
    types.ModuleType("src.shared.python.logging_pkg.logging_config"),
)
_logging_config_stub.get_logger = logging.getLogger  # type: ignore[attr-defined]
_logging_config_stub.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def vcr_config() -> dict[str, object]:
    """Shared VCR config for all integration test modules.

    Returns:
        A dict consumed by ``pytest-recording`` to redact auth headers
        and force replay-only mode in CI.
    """
    return {
        "filter_headers": [
            ("authorization", "REDACTED"),
            ("x-api-key", "REDACTED"),
            ("cookie", "REDACTED"),
            ("notion-version", "2022-06-28"),
        ],
        "filter_query_parameters": [
            ("api_key", "REDACTED"),
            ("token", "REDACTED"),
        ],
        "decode_compressed_response": True,
        "record_mode": "none",
        "match_on": ["method", "scheme", "host", "path"],
    }
