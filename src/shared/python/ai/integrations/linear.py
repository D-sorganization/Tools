"""Linear integration tools for Sidekick."""

from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from typing import Any

from src.shared.python.ai.tool_registry import ToolCategory, get_global_registry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class LinearError(RuntimeError):
    """Base exception for all Linear API errors."""


class LinearRateLimitError(LinearError):
    """Raised when Linear returns HTTP 429 and retries are exhausted."""


class LinearAuthError(LinearError):
    """Raised immediately on HTTP 401 or 403 (no retry)."""


class LinearNetworkError(LinearError):
    """Raised on network-level failures (connection refused, timeout, etc.)."""


# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------

_LINEAR_API_TOKEN: str | None = None
_LINEAR_API_URL = "https://api.linear.app/graphql"

_MAX_RETRIES = 3
_MAX_PAGES = 10


def set_linear_api_token(token: str) -> None:
    """Store the Linear API token in memory for session use."""
    global _LINEAR_API_TOKEN
    _LINEAR_API_TOKEN = token


# ---------------------------------------------------------------------------
# GraphQL queries
# ---------------------------------------------------------------------------

_ISSUES_QUERY = """
query Issues($filter: IssueFilter, $after: String) {
  issues(filter: $filter, after: $after, first: 50) {
    pageInfo {
      endCursor
      hasNextPage
    }
    nodes {
      id
      title
      state {
        name
      }
      url
    }
  }
}
"""

_CREATE_ISSUE_MUTATION = """
mutation IssueCreate($input: IssueCreateInput!) {
  issueCreate(input: $input) {
    success
    issue {
      id
      title
      url
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------


def _run_linear_query(
    query_str: str,
    variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a single GraphQL request against the Linear API.

    Retries on 429 (rate-limit) and 5xx (server error) with backoff.
    Raises immediately on 401/403 (auth errors).

    Returns:
        The parsed JSON response body.

    Raises:
        LinearAuthError: On HTTP 401 or 403.
        LinearRateLimitError: On persistent HTTP 429 after retries.
        LinearError: On persistent HTTP 5xx after retries.
        LinearNetworkError: On network-level failure.
    """
    import json as _json

    if not _LINEAR_API_TOKEN:
        raise LinearAuthError(
            "Linear API token is not configured. Call set_linear_api_token() first."
        )

    payload = _json.dumps({"query": query_str, "variables": variables or {}}).encode(
        "utf-8"
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": _LINEAR_API_TOKEN,
    }

    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        req = urllib.request.Request(
            _LINEAR_API_URL, data=payload, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                body = _json.loads(resp.read().decode("utf-8"))
                if "errors" in body:
                    messages = "; ".join(
                        e.get("message", str(e)) for e in body["errors"]
                    )
                    raise LinearError(f"Linear GraphQL error: {messages}")
                return body

        except urllib.error.HTTPError as exc:
            status = exc.code

            if status in (401, 403):
                raise LinearAuthError(
                    f"Linear authentication failed (HTTP {status})"
                ) from exc

            if status == 429:
                retry_after_str = exc.headers.get("Retry-After", "")
                try:
                    wait = float(retry_after_str)
                except (ValueError, TypeError):
                    wait = 60.0
                logger.warning(
                    "Linear rate-limited (429). Waiting %.0fs (attempt %d/%d).",
                    wait,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)
                    last_exc = exc
                    continue
                raise LinearRateLimitError(
                    f"Linear rate limit exceeded after {_MAX_RETRIES} retries"
                ) from exc

            if 500 <= status < 600:
                wait = 2.0 ** (attempt + 1)  # 2, 4, 8
                logger.warning(
                    "Linear server error (HTTP %d). Waiting %.0fs (attempt %d/%d).",
                    status,
                    wait,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)
                    last_exc = exc
                    continue
                raise LinearError(
                    f"Linear server error (HTTP {status}) after {_MAX_RETRIES} retries"
                ) from exc

            # Other HTTP errors (4xx) — raise immediately
            raise LinearError(f"Linear HTTP error: {status}") from exc

        except urllib.error.URLError as exc:
            raise LinearNetworkError(
                f"Network error contacting Linear: {exc.reason}"
            ) from exc

    # Should be unreachable, but satisfy type checker
    if last_exc is not None:
        raise LinearError("Linear request failed after retries") from last_exc
    raise LinearError("Linear request failed")  # pragma: no cover


def _run_paginated_query(
    query_str: str,
    variables: dict[str, Any] | None = None,
    *,
    page_key: str,
    max_pages: int = _MAX_PAGES,
) -> list[dict[str, Any]]:
    """Execute a paginated GraphQL query, following cursors until exhausted.

    Args:
        query_str: GraphQL query that accepts an ``after`` variable and
            includes ``pageInfo { endCursor hasNextPage }`` in the
            ``page_key`` field of ``data``.
        variables: Additional query variables (``after`` is injected here).
        page_key: Top-level key in ``data`` that contains ``pageInfo``
            and ``nodes`` (e.g. ``"issues"``).
        max_pages: Hard cap on pages fetched (default 10).

    Returns:
        Flat list of all ``nodes`` across all pages.
    """
    vars_: dict[str, Any] = dict(variables or {})
    all_nodes: list[dict[str, Any]] = []

    for page_num in range(max_pages):
        response = _run_linear_query(query_str, vars_)
        data_block = response.get("data", {}).get(page_key, {})
        nodes = data_block.get("nodes", [])
        all_nodes.extend(nodes)

        page_info = data_block.get("pageInfo", {})
        has_next = page_info.get("hasNextPage", False)
        end_cursor = page_info.get("endCursor")

        if not has_next or not end_cursor:
            break

        if page_num + 1 >= max_pages:
            logger.warning(
                "Linear pagination stopped at max_pages=%d guard.", max_pages
            )
            break

        vars_["after"] = end_cursor

    return all_nodes


# ---------------------------------------------------------------------------
# Registered tools
# ---------------------------------------------------------------------------

registry = get_global_registry()


@registry.register(
    "linear_query_issues",
    "Query Linear issues based on a search term or status.",
    category=ToolCategory.ANALYSIS,
)
def linear_query_issues(query: str, status: str = "open") -> dict[str, Any]:
    """Query Linear for issues with cursor-based pagination.

    Args:
        query: The search term to find relevant issues.
        status: The status of issues to return (e.g. 'open', 'done').
    """
    if not _LINEAR_API_TOKEN:
        return {
            "error": "Linear API token is not configured. Please provide it in settings."  # noqa: E501
        }

    logger.info("Querying Linear for '%s' with status '%s'", query, status)

    variables: dict[str, Any] = {
        "filter": {
            "and": [
                {"title": {"containsIgnoreCase": query}},
                {"state": {"name": {"eqIgnoreCase": status}}},
            ]
        }
    }

    try:
        nodes = _run_paginated_query(_ISSUES_QUERY, variables, page_key="issues")
    except LinearError as exc:
        return {"error": str(exc)}

    issues = [
        {
            "id": n.get("id", ""),
            "title": n.get("title", ""),
            "status": n.get("state", {}).get("name", status),
            "url": n.get("url", ""),
        }
        for n in nodes
    ]
    return {"success": True, "issues": issues}


@registry.register(
    "linear_create_issue",
    "Create a new issue in Linear.",
    category=ToolCategory.ANALYSIS,
    requires_confirmation=True,
)
def linear_create_issue(
    title: str, description: str, team_id: str = ""
) -> dict[str, Any]:
    """Create a new Linear issue.

    Args:
        title: The title of the new issue.
        description: Detailed description in Markdown.
        team_id: The team ID to create the issue under.
    """
    if not _LINEAR_API_TOKEN:
        return {
            "error": "Linear API token is not configured. Please provide it in settings."  # noqa: E501
        }

    logger.info("Creating Linear issue: %s", title)

    input_: dict[str, Any] = {"title": title, "description": description}
    if team_id:
        input_["teamId"] = team_id

    try:
        response = _run_linear_query(_CREATE_ISSUE_MUTATION, {"input": input_})
    except LinearError as exc:
        return {"error": str(exc)}

    result = response.get("data", {}).get("issueCreate", {})
    issue = result.get("issue", {})
    return {
        "success": result.get("success", False),
        "issue": {
            "id": issue.get("id", ""),
            "title": issue.get("title", title),
            "url": issue.get("url", ""),
        },
    }
