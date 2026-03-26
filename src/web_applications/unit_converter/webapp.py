"""Unit Converter Flask web application.

Provides a REST API for unit conversions with a theme system that inherits
from the shared theme-definitions used by all PyQt6 and web applications.
"""

from __future__ import annotations

import logging
from typing import Any

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

from .converter import UnitConverter
from .web_theme import (
    all_themes_as_css,
    get_default_theme_name,
    get_themes_for_api,
)

logger = logging.getLogger(__name__)


def _add_security_headers(response: Response) -> Response:
    """Add security headers to every response."""
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; "
        "img-src 'self' data:; "
        "object-src 'none'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self';"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"  # noqa: E501
    )
    response.headers["Permissions-Policy"] = (
        "geolocation=(), camera=(), microphone=(), payment=(), usb=()"
    )
    return response


def _parse_convert_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse and validate the conversion API payload.

    Returns:
        Dict with parsed value, from_unit, to_unit, and optional gas params.

    Raises:
        ValueError: If required fields are missing.
    """
    value = float(payload.get("value", 0))
    from_unit = str(payload.get("from_unit", "")).strip()
    to_unit = str(payload.get("to_unit", "")).strip()

    if not from_unit or not to_unit:
        raise ValueError("Both from_unit and to_unit are required")

    temperature = payload.get("temperature")
    pressure = payload.get("pressure")
    gas_density_stp = payload.get("gas_density_stp")

    return {
        "value": value,
        "from_unit": from_unit,
        "to_unit": to_unit,
        "temperature": float(temperature) if temperature is not None else None,
        "pressure": float(pressure) if pressure is not None else None,
        "gas_type": str(payload.get("gas_type", "air")),
        "standard_condition": str(payload.get("standard_condition", "SCFM_60F")),
        "gas_density_stp": (
            float(gas_density_stp) if gas_density_stp is not None else None
        ),  # noqa: E501
    }


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )

    converter = UnitConverter()
    _theme_css = all_themes_as_css()

    app.after_request(_add_security_headers)

    @app.get("/")
    def index() -> str:
        """Render the unit converter page."""
        categories = converter.get_categories()
        category_data = {
            cat: {
                "label": converter.get_category_label(cat),
                "units": converter.get_units_for_category(cat),
            }
            for cat in categories
        }
        return str(
            render_template(
                "index.html",
                categories=categories,
                category_data=category_data,
                themes=get_themes_for_api(),
                default_theme=get_default_theme_name(),
            )
        )

    @app.get("/api/theme.css")
    def api_theme_css() -> Response:
        """Serve dynamically-generated theme CSS from shared themes.json."""
        response = Response(_theme_css, mimetype="text/css")
        response.headers["Cache-Control"] = "public, max-age=3600"
        return response

    @app.get("/api/themes")
    def api_themes() -> tuple[Any, int]:
        """Return list of available themes from the shared theme system."""
        return jsonify(get_themes_for_api()), 200

    @app.post("/api/convert")
    def api_convert() -> tuple[Any, int]:
        """Perform a unit conversion via the API."""
        payload = request.get_json(silent=True) or {}
        try:
            params = _parse_convert_payload(payload)
            result = converter.convert(
                params["value"],
                params["from_unit"],
                params["to_unit"],
                temperature=params["temperature"],
                pressure=params["pressure"],
                gas_type=params["gas_type"],
                standard_condition=params["standard_condition"],
                gas_density_stp=params["gas_density_stp"],
            )
            return (
                jsonify(
                    {
                        "value": result.value,
                        "from_unit": result.from_unit,
                        "to_unit": result.to_unit,
                        "result": result.result,
                        "formatted": converter.format_number(result.result),
                        "category": result.category,
                    }
                ),
                200,
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except (ZeroDivisionError, OverflowError, TypeError):
            logger.exception("Conversion failed")
            return jsonify({"error": "An internal error occurred."}), 500

    @app.get("/api/categories")
    def api_categories() -> tuple[Any, int]:
        """Return available categories and their units."""
        categories = converter.get_categories()
        data = {
            cat: {
                "label": converter.get_category_label(cat),
                "units": converter.get_units_for_category(cat),
            }
            for cat in categories
        }
        return jsonify(data), 200

    @app.get("/api/units/<category>")
    def api_units(category: str) -> tuple[Any, int]:
        """Return units for a specific category."""
        units = converter.get_units_for_category(category)
        if not units:
            return jsonify({"error": f"Unknown category: {category}"}), 404
        return (
            jsonify(
                {
                    "category": category,
                    "label": converter.get_category_label(category),
                    "units": units,
                }
            ),
            200,
        )

    @app.get("/manifest.json")
    def manifest() -> Response:
        """Serve the PWA manifest."""
        return send_from_directory(app.static_folder or "static", "manifest.json")

    return app
