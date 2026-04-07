"""FastAPI adapter for the model_generation REST API."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from .rest_api_routes import APIRequest, HTTPMethod, ModelGenerationAPI, Route

logger = logging.getLogger(__name__)


def _request_uses_form_parsing(request: Request) -> bool:
    """Return True when the incoming request may contain form-backed uploads."""
    content_type = request.headers.get("content-type", "").lower()
    return (
        "multipart/form-data" in content_type
        or "application/x-www-form-urlencoded" in content_type
    )


class FastAPIAdapter:
    """Adapter for FastAPI framework."""

    def __init__(self, api: ModelGenerationAPI) -> None:
        if not (api is not None):
            raise ValueError("api must be provided")
        self.api = api

    def register(self, app: Any) -> None:
        """Register routes with FastAPI app."""
        for route in self.api.get_routes():

            def make_handler(r: Route) -> Any:
                async def handler(request: Request) -> Any:
                    body = None
                    try:
                        body = await request.json()
                    except (ValueError, UnicodeDecodeError) as exc:
                        logger.debug("Failed to parse request JSON body: %s", exc)

                    files: dict[str, bytes] = {}
                    if _request_uses_form_parsing(request):
                        form = await request.form()
                        for key, value in form.items():
                            if hasattr(value, "read"):
                                files[key] = await value.read()

                    api_request = APIRequest(
                        method=HTTPMethod(request.method),
                        path=request.url.path,
                        query_params={
                            **dict(request.query_params),
                            **dict(request.path_params),
                        },
                        body=body,
                        files=files,
                        headers=dict(request.headers),
                    )

                    response = self.api.handle_request(api_request)

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

            app.add_api_route(
                route.path,
                make_handler(route),
                methods=[route.method.value],
                tags=route.tags,
                summary=route.description,
            )
