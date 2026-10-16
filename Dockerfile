FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/opt/hf

WORKDIR /app

# 1) CPU-only torch. The default wheel on Linux pulls several GB of CUDA libraries.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2) Runtime dependencies only (no mlflow, dvc, ragas or pytest).
#    A stub "src" package lets pip install the dependencies before the real code
#    is copied, so this layer is cached when only the code changes.
COPY pyproject.toml ./
RUN mkdir src && touch src/__init__.py \
    && pip install . \
    && pip uninstall -y legal-rag

# 3) Application code.
COPY src ./src

# 4) Bake the embedding model into the image, using the name from src/config.py.
RUN python -c "from sentence_transformers import SentenceTransformer; from src.config import EMBEDDING_MODEL; SentenceTransformer(EMBEDDING_MODEL)"

# 5) Bake the vector store (built by `dvc repro` or fetched by `dvc pull`).
COPY data/processed/vector_store ./data/processed/vector_store

# 6) Fail the build early if the app cannot be imported.
RUN python -c "from src.api import app; print('API imports OK')"

# The model is already inside the image, so never try to reach the internet at runtime.
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]