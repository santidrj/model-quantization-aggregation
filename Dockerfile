# -----------------------------------------------------------------------------
# Stage 1: Build stage - Install dependencies with uv
# -----------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy only dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv sync (uses lock file for reproducibility)
# Using --no-dev to exclude development dependencies
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
RUN uv sync --no-dev

# -----------------------------------------------------------------------------
# Stage 2: Runtime stage - Minimal image with only runtime requirements
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    ROOT="/app"

# Install Times New Roman font
# 1. Enable contrib repository for ttf-mscorefonts-installer
# 2. Accept EULA automatically
# 3. Install fonts and fontconfig
RUN sed -i 's/Components: main/Components: main contrib/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections && \
    apt-get install -y --no-install-recommends ttf-mscorefonts-installer fontconfig && \
    fc-cache -f && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy project files
COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY data/ ./data/
COPY pyproject.toml ./
COPY figures.mplstyle ./

# Create reports directory for output
RUN mkdir -p reports/figures

# Expose Jupyter notebook port
EXPOSE 8888

# Default command: Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
