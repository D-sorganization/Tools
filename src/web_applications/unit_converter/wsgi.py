"""WSGI entry point for the Unit Converter web application."""

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
