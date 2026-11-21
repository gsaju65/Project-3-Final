# Drone Object Classifier: one Dockerfile, two images.
#   docker build -t drone-clf .                      full project: train, test or serve (default)
#   docker build --target serve -t drone-clf-api .   slim API image with the trained model inside (deployed to Cloud Run)

FROM python:3.12-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt
COPY config.yaml .
COPY src/ src/
COPY api/ api/

# Slim API image for Cloud Run. Needs a trained model in models/ when it is built.
FROM base AS serve
COPY models/ models/
RUN useradd --create-home appuser
USER appuser
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]

# Full project (default target): download the data from GCS, train with MLflow, run the tests, or serve
FROM base AS full
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY scripts/ scripts/
COPY tests/ tests/
COPY pytest.ini .
COPY docker/entrypoint.sh docker/entrypoint.sh
RUN sed -i 's/\r$//' docker/entrypoint.sh
EXPOSE 8080
ENTRYPOINT ["sh", "docker/entrypoint.sh"]
CMD ["train"]