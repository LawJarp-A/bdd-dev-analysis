FROM python:3.10-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src/ src/

# Expose Streamlit port
EXPOSE 8501

# Health check (curl not available in slim image, use Python)
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Run: download data if needed, then launch dashboard
CMD uv run python -m src.download_data && \
    uv run streamlit run src/dashboard.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true
