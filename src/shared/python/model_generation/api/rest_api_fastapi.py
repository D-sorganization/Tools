"""FastAPI adapter for the framework-neutral model_generation API."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from model_generation.api.rest_api_contracts import APIRequest, HTTPMethod, Route
from model_generation.api.rest_api_core import ModelGenerationAPI

logger = logging.getLogger(__name__)


class FastAPIAdapter:
    """Register model_generation routes on a FastAPI application."""

    def __init__(self, api: ModelGenerationAPI) -> None:
        """Bind the adapter to a framework-neutral API instance."""
        if api is None:
            raise ValueError("api must be provided")
        self.api = api

    def register(self, app: Any) -> None:
        """Register all API routes on the FastAPI app."""
        for route in self.api.get_routes():
            app.add_api_route(
                route.path,
                self._make_handler(route),
                methods=[route.method.value],
                tags=route.tags,
                summary=route.description,
            )

    def _make_handler(self, route: Route) -> Callable[..., Any]:
        """Build an async FastAPI handler for a route definition."""
        from fastapi import Request, Response
        from fastapi.responses import JSONResponse

        async def handler(request: Request, **kwargs: Any) -> Any:
            response = self.api.handle_request(
                APIRequest(
                    method=HTTPMethod(request.method),
                    path=request.url.path,
                    query_params={**request.query_params, **kwargs},
                    body=await self._json_body(request),
                    files=await self._uploaded_files(request),
                    headers=dict(request.headers),
                )
            )
            if isinstance(response.body, bytes):
                return Response(
                    content=response.body,
                    status_code=response.status_code,
                    media_type=response.content_type,
                    headers=response.headers,
                )
            return JSONResponse(
                content=response.body,
                status_code=response.status_code,
                headers=response.headers,
            )

        return handler

    async def _json_body(self, request: Any) -> dict[str, Any] | None:
        """Parse a JSON body when present, tolerating non-JSON requests."""
        try:
            return await request.json()
        except (ValueError, UnicodeDecodeError) as error:
            logger.debug("Failed to parse request JSON body: %s", error)
            return None

    async def _uploaded_files(self, request: Any) -> dict[str, bytes]:
        """Collect uploaded file payloads from a FastAPI form body."""
        files: dict[str, bytes] = {}
        for key, value in (await request.form()).items():
            if hasattr(value, "read"):
                files[key] = await value.read()
        return files
