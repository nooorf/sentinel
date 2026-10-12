# syntax=docker/dockerfile:1
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 SENTINEL_DATA_DIR=/app/var \
    PIP_DEFAULT_TIMEOUT=120 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# pip's download cache lives in a BuildKit cache mount (not in the image), so a failed or repeated build
# reuses every wheel already downloaded; long timeout + retries ride out slow connections
RUN --mount=type=cache,target=/root/.cache/pip pip install --retries 10 -r requirements.txt
COPY app ./app
COPY scripts ./scripts
RUN useradd -m -u 1000 sentinel && mkdir -p /app/var /app/data /app/reports && chown -R sentinel /app
USER sentinel
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -fs http://localhost:8000/health || exit 1
CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
