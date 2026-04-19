"""WSGI entry point for the Unit Converter web application.

Production deployments should use a WSGI server (e.g. gunicorn):
    gunicorn 'src.web_applications.unit_converter.wsgi:app'

The __main__ block is for local development only. Set UNIT_CONVERTER_DEBUG=1 to
enable the Werkzeug interactive debugger; it must never be enabled in
production as it exposes an RCE surface on any unhandled exception.
"""

import os

from .webapp import create_app

app = create_app()


def _is_debug_enabled() -> bool:
    """Return True when debug execution is explicitly enabled."""
    return os.getenv("UNIT_CONVERTER_DEBUG", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


if __name__ == "__main__":
    app.run(debug=_is_debug_enabled(), port=5001)
