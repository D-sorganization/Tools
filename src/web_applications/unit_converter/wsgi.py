"""WSGI entry point for the Unit Converter web application.

Production deployments should use a WSGI server (e.g. gunicorn):
    gunicorn 'src.web_applications.unit_converter.wsgi:app'

The __main__ block is for local development only. Set FLASK_DEBUG=1 to
enable the Werkzeug interactive debugger; it must never be enabled in
production as it exposes an RCE surface on any unhandled exception.
"""

import os

from .webapp import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5001)
