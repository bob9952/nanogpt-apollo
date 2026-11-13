FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1

WORKDIR /app

RUN python -m pip install --no-cache-dir uv==0.11.2

COPY pyproject.toml uv.lock ./
COPY apollo_rag/ ./apollo_rag/

RUN uv sync --locked --no-dev --no-editable

ENTRYPOINT ["/app/.venv/bin/python", "-m", "apollo_rag"]
