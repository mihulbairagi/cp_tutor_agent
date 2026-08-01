# syntax=docker/dockerfile:1
#
# Two things drive this file: layer caching, and not running as root.
#
# Dependencies are installed in their own layer before the source is copied, so
# editing a Python file rebuilds in seconds instead of re-resolving pip. The
# runtime user is unprivileged, and the data directory is chowned to it because
# the SQLite cache and the Chroma index are written at runtime.
#
# The image is CPU-only on purpose. sentence-transformers pulls in torch, whose
# default wheel carries ~2 GB of CUDA libraries that are dead weight in a
# container that will never see a GPU; the CPU index cuts the image by more than
# half.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Cache the embedding model inside the image rather than downloading it on
    # first request, which would make the first /sync mysteriously slow.
    HF_HOME=/home/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/app/.cache/huggingface

# curl is needed by the compose healthcheck; nothing else is added.
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /usr/sbin/nologin app

WORKDIR /app

COPY requirements.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY app ./app
COPY eval ./eval

RUN mkdir -p /app/data && chown -R app:app /app /home/app

USER app

# Pre-download the embedding model so the first request is not paying for it.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
