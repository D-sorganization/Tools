"""Flask adapter for the framework-neutral model_generation API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from model_generation.api.rest_api_contracts import APIRequest, HTTPMethod, Route
from model_generation.api.rest_api_core import ModelGenerationAPI


class FlaskAdapter:
    """Register model_generation routes on a Flask application."""

    def __init__(self, api: ModelGenerationAPI) -> None:
        """Bind the adapter to a framework-neutral API instance."""
        if api is None:
            raise ValueError("api must be provided")
        self.api = api

    def register(self, app: Any) -> None:
        """Register all API routes on the Flask app."""
        for route in self.api.get_routes():
            app.add_url_rule(
                self._flask_path(route),
                endpoint=self._endpoint_name(route),
                view_func=self._make_handler(route),
                methods=[route.method.value],
            )

    def _make_handler(self, route: Route) -> Callable[..., Any]:
        """Build a Flask view function for a route definition."""
        from flask import jsonify, make_response
        from flask import request as flask_request

        def handler(**kwargs: Any) -> Any:
            response = self.api.handle_request(
                APIRequest(
                    method=HTTPMethod(flask_request.method),
                    path=flask_request.path,
                    query_params={**flask_request.args, **kwargs},
                    body=flask_request.get_json(silent=True),
                    files={
                        name: upload.read()
                        for name, upload in flask_request.files.items()
                    },
                    headers=dict(flask_request.headers),
                )
            )
            return self._make_response(
                response=response,
                jsonify=jsonify,
                make_response=make_response,
            )

        return handler

    def _make_response(self, *, response: Any, jsonify: Any, make_response: Any) -> Any:
        """Translate an APIResponse into Flask's response type."""
        body = (
            response.body
            if not isinstance(response.body, dict)
            else jsonify(response.body)
        )
        flask_response = make_response(body)
        flask_response.status_code = response.status_code
        flask_response.content_type = response.content_type
        for header_name, header_value in response.headers.items():
            flask_response.headers[header_name] = header_value
        return flask_response

    def _endpoint_name(self, route: Route) -> str:
        """Return a stable Flask endpoint name for a route."""
        return route.path.replace("/", "_").replace("{", "").replace("}", "")

    def _flask_path(self, route: Route) -> str:
        """Convert route parameter syntax from {param} to <param>."""
        return route.path.replace("{", "<").replace("}", ">")
