# Product matching API. Models are built inside the image from the CSV and labels.
#   docker build -t product-matching .
#   docker run --rm -p 8000:8000 product-matching
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    PM_ROOT=/app \
    PM_ARTIFACTS=/app/artifacts

WORKDIR /app

# CPU-only torch first (the default PyPI wheel on Linux pulls in several GB of CUDA).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

COPY data ./data
COPY labels ./labels
# Downloads the embedding model and fits every component (a few minutes on CPU).
RUN pm build && rm -rf /app/artifacts/embeddings

RUN useradd --create-home --uid 1000 app && chown -R app /app /opt/hf
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "product_matching.api:app", "--host", "0.0.0.0", "--port", "8000"]
