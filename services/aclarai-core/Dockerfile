# aclarai Core Service - Main Processing Engine
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables for Python
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv, the fast Python package installer
RUN pip install uv

# Copy the entire project context into the image.
# This makes the root pyproject.toml and all shared/service code available.
COPY . .

# Install the specific dependencies from this subproject's pyproject.toml
RUN uv pip install --system -e ./services/aclarai-core


# Install default prompt templates to /settings/prompts (make them available for user customization)
# /settings is mounted to the host, so users can access and modify these prompts
RUN mkdir -p /settings/prompts && python services/aclarai-core/install/install_prompts.py --all --prompts-dir /settings/prompts

# Create a non-root user for security
RUN useradd -m -u 1000 aclarai && chown -R aclarai:aclarai /app
USER aclarai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import aclarai_core; print('OK')" || exit 1

# Run the application
CMD ["python", "-m", "aclarai_core.main"]