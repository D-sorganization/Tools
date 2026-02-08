"""WSGI entry point for the Unit Converter web application."""

from .webapp import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5001)
