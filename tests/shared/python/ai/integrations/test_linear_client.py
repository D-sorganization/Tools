"""Tests for the Linear GraphQL client (Phase 2): pagination, retry, error hierarchy.

Issue #2830.
"""

from __future__ import annotations

import json
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — one level deeper than tests/shared/python/ai/ which uses
# parents[4], so we use parents[5].
# ---------------------------------------------------------------------------
import types
import urllib.error
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_TOOLS_ROOT = Path(__file__).resolve().parents[5]
if str(_TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TOOLS_ROOT))

# Stub the transitive dependency chain so linear.py can be imported without
# pulling in the full ai/__init__.py, logging_pkg, etc.
_STUBS: list[tuple[str, str | None]] = [
    ("src", "src"),
    ("src.shared", "src/shared"),
    ("src.shared.python", "src/shared/python"),
    ("src.shared.python.ai", "src/shared/python/ai"),
    ("src.shared.python.ai.integrations", "src/shared/python/ai/integrations"),
    ("src.shared.python.logging_pkg", None),
    ("src.shared.python.logging_pkg.logging_config", None),
]
for _mod_name, _rel_path in _STUBS:
    if _mod_name not in sys.modules:
        _stub = types.ModuleType(_mod_name)
        if _rel_path is not None:
            _stub.__path__ = [str(_TOOLS_ROOT / _rel_path)]  # type: ignore[attr-defined]
        sys.modules[_mod_name] = _stub

# Provide get_logger so tool_registry.py can import it.
_log_cfg = sys.modules["src.shared.python.logging_pkg.logging_config"]
import logging as _logging

_log_cfg.get_logger = _logging.getLogger  # type: ignore[attr-defined]
_log_cfg.setup_logging = lambda *a, **kw: None  # type: ignore[attr-defined]

# Stub tool_registry with enough surface for linear.py's @registry.register usage.
_tr_stub = types.ModuleType("src.shared.python.ai.tool_registry")
_tr_stub.__path__ = []  # type: ignore[attr-defined]


class _ToolCategory:
    ANALYSIS = "ANALYSIS"


class _FakeRegistry:
    def register(self, name, description, category=None, requires_confirmation=False):
        def decorator(fn):
            return fn

        return decorator


_tr_stub.ToolCategory = _ToolCategory  # type: ignore[attr-defined]
_tr_stub.get_global_registry = _FakeRegistry  # type: ignore[attr-defined]
sys.modules["src.shared.python.ai.tool_registry"] = _tr_stub

# Stub exceptions + types used by tool_registry.py internals (not needed by
# linear.py itself, but tool_registry.py is imported transitively).
for _extra in (
    "src.shared.python.ai.exceptions",
    "src.shared.python.ai.types",
):
    if _extra not in sys.modules:
        sys.modules[_extra] = types.ModuleType(_extra)

# ---------------------------------------------------------------------------
# Subject under test
# ---------------------------------------------------------------------------
from src.shared.python.ai.integrations.linear import (  # noqa: E402
    LinearAuthError,
    LinearError,
    LinearNetworkError,
    LinearRateLimitError,
    _run_linear_query,
    _run_paginated_query,
    linear_query_issues,
    set_linear_api_token,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOKEN = "lin_api_test_token"


def _make_http_error(
    status: int, headers: dict[str, str] | None = None
) -> urllib.error.HTTPError:
    """Build a urllib.error.HTTPError with optional headers dict."""
    hdrs = MagicMock()
    hdrs.get = lambda key, default=None: (headers or {}).get(key, default)
    return urllib.error.HTTPError(
        url="https://api.linear.app/graphql",
        code=status,
        msg=f"HTTP {status}",
        hdrs=hdrs,
        fp=None,
    )


def _mock_urlopen_response(body: dict[str, Any]):
    """Return a context-manager mock that yields a fake HTTP response."""
    encoded = json.dumps(body).encode("utf-8")
    fake_resp = MagicMock()
    fake_resp.read.return_value = encoded
    fake_resp.__enter__ = lambda s: s
    fake_resp.__exit__ = MagicMock(return_value=False)
    return fake_resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_token(monkeypatch):
    """Ensure the module-level token is set for every test."""
    set_linear_api_token(_TOKEN)
    yield
    # Reset after test
    set_linear_api_token("")


# ---------------------------------------------------------------------------
# 1. Error hierarchy
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    def test_linear_rate_limit_is_linear_error(self):
        assert issubclass(LinearRateLimitError, LinearError)

    def test_linear_auth_is_linear_error(self):
        assert issubclass(LinearAuthError, LinearError)

    def test_linear_network_is_linear_error(self):
        assert issubclass(LinearNetworkError, LinearError)

    def test_linear_error_is_runtime_error(self):
        assert issubclass(LinearError, RuntimeError)


# ---------------------------------------------------------------------------
# 2. Auth errors — immediate raise, no retry
# ---------------------------------------------------------------------------


class TestAuthErrors:
    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_error_raised_immediately(self, status: int):
        with patch("urllib.request.urlopen", side_effect=_make_http_error(status)):
            with pytest.raises(LinearAuthError):
                _run_linear_query("{ issues { nodes { id } } }")

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_error_no_sleep(self, status: int):
        with (
            patch("urllib.request.urlopen", side_effect=_make_http_error(status)),
            patch("time.sleep") as mock_sleep,
        ):
            with pytest.raises(LinearAuthError):
                _run_linear_query("{ issues { nodes { id } } }")
        mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Rate-limit retry (429)
# ---------------------------------------------------------------------------


class TestRateLimitRetry:
    def test_429_triggers_sleep_with_retry_after(self):
        """429 with Retry-After header → sleep that many seconds."""
        error_429 = _make_http_error(429, headers={"Retry-After": "30"})
        good_response = _mock_urlopen_response(
            {
                "data": {
                    "issues": {
                        "nodes": [{"id": "ENG-1"}],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
        )

        side_effects = [error_429, good_response]

        with (
            patch("urllib.request.urlopen", side_effect=side_effects),
            patch("time.sleep") as mock_sleep,
        ):
            result = _run_linear_query("{ issues { nodes { id } } }")

        mock_sleep.assert_called_once_with(30.0)
        assert result is not None

    def test_429_default_wait_when_no_retry_after(self):
        """429 without Retry-After header → sleep 60 seconds."""
        error_429 = _make_http_error(429, headers={})
        good_response = _mock_urlopen_response({"data": {}})

        with (
            patch("urllib.request.urlopen", side_effect=[error_429, good_response]),
            patch("time.sleep") as mock_sleep,
        ):
            _run_linear_query("{ issues { nodes { id } } }")

        mock_sleep.assert_called_once_with(60.0)

    def test_429_exhausted_raises_rate_limit_error(self):
        """Persistent 429 after all retries → LinearRateLimitError."""
        errors = [_make_http_error(429, headers={"Retry-After": "1"})] * 4

        with (
            patch("urllib.request.urlopen", side_effect=errors),
            patch("time.sleep"),
        ):
            with pytest.raises(LinearRateLimitError):
                _run_linear_query("{ issues { nodes { id } } }")

    def test_429_retry_count(self):
        """Confirm exactly _MAX_RETRIES sleep calls before giving up."""
        errors = [_make_http_error(429, headers={"Retry-After": "1"})] * 4

        with (
            patch("urllib.request.urlopen", side_effect=errors),
            patch("time.sleep") as mock_sleep,
        ):
            with pytest.raises(LinearRateLimitError):
                _run_linear_query("{ issues { nodes { id } } }")

        # 3 retries means 3 sleeps
        assert mock_sleep.call_count == 3


# ---------------------------------------------------------------------------
# 4. 5xx exponential backoff
# ---------------------------------------------------------------------------


class TestServerErrorRetry:
    def test_5xx_exponential_backoff(self):
        """5xx uses exponential 2/4/8 backoff then raises LinearError."""
        errors = [_make_http_error(503)] * 4

        with (
            patch("urllib.request.urlopen", side_effect=errors),
            patch("time.sleep") as mock_sleep,
        ):
            with pytest.raises(LinearError):
                _run_linear_query("{ issues { nodes { id } } }")

        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_calls == [2.0, 4.0, 8.0]

    def test_5xx_recovers_on_second_attempt(self):
        """5xx on first attempt → sleep → success on second."""
        good_response = _mock_urlopen_response({"data": {"ok": True}})
        side_effects = [_make_http_error(500), good_response]

        with (
            patch("urllib.request.urlopen", side_effect=side_effects),
            patch("time.sleep") as mock_sleep,
        ):
            result = _run_linear_query("{ ok }")

        mock_sleep.assert_called_once_with(2.0)
        assert result == {"data": {"ok": True}}


# ---------------------------------------------------------------------------
# 5. Cursor-based pagination
# ---------------------------------------------------------------------------


def _page_response(nodes: list[dict], *, has_next: bool, cursor: str | None) -> dict:
    return {
        "data": {
            "issues": {
                "nodes": nodes,
                "pageInfo": {
                    "hasNextPage": has_next,
                    "endCursor": cursor,
                },
            }
        }
    }


_PAGINATED_Q = (
    "query Issues($after: String) { "
    "issues(after: $after) { "
    "pageInfo { endCursor hasNextPage } "
    "nodes { id title } } }"
)


class TestPagination:
    def test_two_page_cursor_pagination(self):
        """Pagination follows cursor through two pages and merges nodes."""
        page1 = _page_response(
            [{"id": "ENG-1", "title": "First"}], has_next=True, cursor="cursor_abc"
        )
        page2 = _page_response(
            [{"id": "ENG-2", "title": "Second"}], has_next=False, cursor=None
        )

        responses = [
            _mock_urlopen_response(page1),
            _mock_urlopen_response(page2),
        ]

        with patch("urllib.request.urlopen", side_effect=responses):
            nodes = _run_paginated_query(_PAGINATED_Q, {}, page_key="issues")

        assert len(nodes) == 2
        assert nodes[0]["id"] == "ENG-1"
        assert nodes[1]["id"] == "ENG-2"

    def test_second_request_uses_cursor(self):
        """The ``after`` variable is set to the cursor from page 1."""
        page1 = _page_response([{"id": "ENG-1"}], has_next=True, cursor="CURSOR_XYZ")
        page2 = _page_response([{"id": "ENG-2"}], has_next=False, cursor=None)

        responses = [
            _mock_urlopen_response(page1),
            _mock_urlopen_response(page2),
        ]

        captured_payloads: list[dict] = []

        def fake_urlopen(req, timeout=30):
            captured_payloads.append(json.loads(req.data.decode("utf-8")))
            return responses.pop(0)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            _run_paginated_query(_PAGINATED_Q, {}, page_key="issues")

        first_vars = captured_payloads[0]["variables"]
        assert first_vars.get("after") is None or "after" not in first_vars
        assert captured_payloads[1]["variables"]["after"] == "CURSOR_XYZ"

    def test_max_pages_guard(self):
        """Pagination stops after max_pages pages even if hasNextPage is True."""
        call_count = 0

        def counting_urlopen(req, timeout=30):
            nonlocal call_count
            call_count += 1
            return _mock_urlopen_response(
                _page_response(
                    [{"id": f"ENG-{call_count}"}], has_next=True, cursor="next"
                )
            )

        with patch("urllib.request.urlopen", side_effect=counting_urlopen):
            nodes = _run_paginated_query(
                _PAGINATED_Q, {}, page_key="issues", max_pages=10
            )

        assert call_count == 10
        assert len(nodes) == 10

    def test_single_page_no_pagination(self):
        """When hasNextPage is False on page 1, only one request is made."""
        page1 = _page_response([{"id": "ENG-1"}], has_next=False, cursor=None)

        call_count = 0

        def counting_urlopen(req, timeout=30):
            nonlocal call_count
            call_count += 1
            return _mock_urlopen_response(page1)

        with patch("urllib.request.urlopen", side_effect=counting_urlopen):
            nodes = _run_paginated_query(_PAGINATED_Q, {}, page_key="issues")

        assert call_count == 1
        assert len(nodes) == 1


# ---------------------------------------------------------------------------
# 6. linear_query_issues integration
# ---------------------------------------------------------------------------


class TestLinearQueryIssues:
    def test_returns_issues_from_paginated_response(self):
        issue_node = {
            "id": "ENG-10",
            "title": "Fix login",
            "state": {"name": "open"},
            "url": "https://linear.app/ENG-10",
        }
        page = _page_response([issue_node], has_next=False, cursor=None)

        with patch("urllib.request.urlopen", return_value=_mock_urlopen_response(page)):
            result = linear_query_issues("login", status="open")

        assert result["success"] is True
        assert len(result["issues"]) == 1
        assert result["issues"][0]["id"] == "ENG-10"

    def test_returns_error_dict_when_no_token(self):
        set_linear_api_token("")
        result = linear_query_issues("anything")
        assert "error" in result

    def test_returns_error_dict_on_auth_failure(self):
        with patch("urllib.request.urlopen", side_effect=_make_http_error(401)):
            result = linear_query_issues("anything")
        assert "error" in result


# ---------------------------------------------------------------------------
# 7. Network error
# ---------------------------------------------------------------------------


class TestNetworkError:
    def test_url_error_raises_linear_network_error(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(LinearNetworkError):
                _run_linear_query("{ issues { nodes { id } } }")
