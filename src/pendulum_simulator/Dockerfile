FROM python:3.12-slim

WORKDIR /app

# System deps for PyQt6 (headless)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 libgl1-mesa-glx libglib2.0-0 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

# Default: run tests
CMD ["pytest", "tests/", "-v", "--tb=short"]
