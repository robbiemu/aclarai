# aclarai Vault Watcher Service - Monitors vault for file changes
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
    inotify-tools \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv, the fast Python package installer
RUN pip install uv

# Copy the entire project context into the image.
# This makes the root pyproject.toml and all shared/service code available.
COPY . .

# Install the UI-specific dependencies from its own pyproject.toml
RUN uv pip install --system -e ./services/vault-watcher

# Create a non-root user for security
RUN useradd -m -u 1000 aclarai && chown -R aclarai:aclarai /app
USER aclarai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import aclarai_vault_watcher; print('OK')" || exit 1

# Run the application
CMD ["python", "-m", "aclarai_vault_watcher.main"]