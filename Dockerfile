# Retrieval API image with the pinned corpus, FAISS index and both models baked in.
# Build: docker build -t steam-review-api .
# Run:   docker run -p 8000:8000 [-e GROQ_API_KEY=...] steam-review-api  ->  http://localhost:8000/docs
ARG PYTHON_IMAGE=python:3.12-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STEAM_RAG_PROJECT_DIR=/app \
    HF_HOME=/app/hf-cache \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

COPY evaluation/benchmark.json ./evaluation/benchmark.json
# Download and verify the pinned artifacts and model revisions once, at build time.
# ensure_artifacts writes through NamedTemporaryFile (mode 0600), so open read access
# for the non-root runtime user afterwards.
RUN python -c "from steam_review_rag.api import load_default_backend; load_default_backend()" \
    && chmod -R a+rX /app/outputs /app/hf-cache

RUN useradd --create-home --uid 1000 api
USER api
# Everything the service needs is in the image; never reach the Hub at runtime.
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "steam_review_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
