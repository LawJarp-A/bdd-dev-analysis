FROM python:3.10-slim

WORKDIR /app

# System deps for opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv and Python deps
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source
COPY src/ src/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Download data if needed, then launch dashboard
CMD uv run python -m src.download_data && \
    uv run streamlit run src/dashboard.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true
