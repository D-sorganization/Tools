FROM python:3.13-slim

WORKDIR /app

# Install standard dependencies directly to keep container fast and lightweight
RUN pip install --no-cache-dir fastapi sqlmodel pymodbus uvicorn websockets

# Copy source code to working directory
COPY . /app

# Expose FastAPI SCADA port
EXPOSE 8000

# Execute server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
