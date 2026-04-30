#!/bin/bash
# Entrypoint script for production container
# Runs migrations, health checks, and starts the Flask app

set -e

echo "[entrypoint] Starting UD Tools application..."
echo "[entrypoint] Python version: $(python3 --version)"
echo "[entrypoint] Flask version: $(python3 -c 'import flask; print(flask.__version__)')"

# Run health check
echo "[entrypoint] Running health checks..."
python3 -c "from web_applications.health_checks import get_health_status; import json; print(json.dumps(get_health_status(), indent=2))"

# Start Flask application
echo "[entrypoint] Starting Flask server on 0.0.0.0:5000..."
exec flask run --host=0.0.0.0
