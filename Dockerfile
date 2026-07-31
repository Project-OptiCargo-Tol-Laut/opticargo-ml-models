FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --gid 10001 opticargo \
    && useradd --uid 10001 --gid 10001 --create-home opticargo

COPY pyproject.toml README.md ./
COPY src ./src
COPY artifacts ./artifacts

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[mlflow]"

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"

CMD ["uvicorn", "opticargo_ml_models.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
