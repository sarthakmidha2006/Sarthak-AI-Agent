# =============================================================================
# AI Persona System ("brain") — production container image
# Base: python:3.11-slim. Installs deps, copies the app, serves via uvicorn.
# =============================================================================
FROM python:3.11-slim AS runtime

# --- Python / pip hygiene -----------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000

WORKDIR /app

# --- System build deps (some wheels need a compiler/headers) ------------------
# Kept minimal; removed after pip install to keep the image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies (cached layer: copy requirements first) --------------
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential

# --- Application source -------------------------------------------------------
COPY app ./app
COPY eval ./eval
COPY scripts ./scripts
COPY pyproject.toml README.md ./

# --- Persisted data (vector store / bm25 / sqlite) ----------------------------
# Mount a volume here in docker-compose to keep the corpus across restarts.
RUN mkdir -p /app/data/resume /app/data/chroma /app/data/bm25
VOLUME ["/app/data"]

EXPOSE 8000

# Health probe hits the FastAPI /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
url='http://127.0.0.1:%s/health' % os.environ.get('API_PORT','8000'); \
sys.exit(0 if urllib.request.urlopen(url, timeout=4).status == 200 else 1)" || exit 1

# Bind host/port from env so the same image works locally and in compose.
CMD ["sh", "-c", "uvicorn app.main:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8000}"]
