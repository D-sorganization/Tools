"""Shared bounded duplicate-rejecting JSON request-body reader."""

from __future__ import annotations

import json

from fastapi import Request

from rate_of_closure.application._workspace_validation import unique_json_object


class StrictHttpFailure(Exception):
    """Expected client error with an HTTP status and public message."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


async def strict_json_document(request: Request, limit: int) -> object:
    """Read one exact JSON object within an explicit byte limit."""
    media_type = request.headers.get("content-type", "").split(";", 1)[0]
    if media_type.strip().lower() != "application/json":
        raise StrictHttpFailure(415, "application/json is required")
    length = request.headers.get("content-length")
    if length is not None:
        try:
            if int(length) > limit:
                raise StrictHttpFailure(413, "request body is too large")
        except ValueError as exc:
            raise StrictHttpFailure(400, "invalid content length") from exc
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise StrictHttpFailure(413, "request body is too large")
    try:
        text = bytes(body).decode("utf-8", errors="strict")
        return json.loads(
            text,
            object_pairs_hook=unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise StrictHttpFailure(400, "invalid JSON request") from exc


__all__ = ["StrictHttpFailure", "strict_json_document"]
