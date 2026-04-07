"""Flask adapter for the model_generation REST API."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .rest_api_routes import APIRequest, HTTPMethod, ModelGenerationAPI, Route


class FlaskAdapter:
    """Adapter for Flask framework."""

    def __init__(self, api: ModelGenerationAPI) -> None:
        if not (api is not None):
            raise ValueError("api must be provided")
        self.api = api

    def register(self, app: Any) -> None:
        """Register routes with Flask app."""
        from flask import jsonify, make_response
        from flask import request as flask_request

        for route in self.api.get_routes():
            endpoint = (
                f"{route.method.value.lower()}_"
                f"{route.path.replace('/', '_').replace('{', '').replace('}', '')}"
            )

            def make_handler(r: Route) -> Callable[..., Any]:
                def handler(**kwargs: Any) -> Any:
                    api_request = APIRequest(
                        method=HTTPMethod(flask_request.method),
                        path=flask_request.path,
                        query_params={**flask_request.args, **kwargs},
                        body=flask_request.get_json(silent=True),
                        files={
                            key: value.read()
                            for key, value in flask_request.files.items()
                        },
                        headers=dict(flask_request.headers),
                    )

                    response = self.api.handle_request(api_request)

                    if isinstance(response.body, bytes):
                        flask_response = make_response(response.body)
                    elif isinstance(response.body, dict):
                        flask_response = make_response(jsonify(response.body))
                    else:
                        flask_response = make_response(response.body)

                    flask_response.status_code = response.status_code
                    flask_response.content_type = response.content_type

                    for key, value in response.headers.items():
                        flask_response.headers[key] = value

                    return flask_response

                return handler

            flask_path = route.path.replace("{", "<").replace("}", ">")
            app.add_url_rule(
                flask_path,
                endpoint=endpoint,
                view_func=make_handler(route),
                methods=[route.method.value],
            )
