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

import pytest


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
