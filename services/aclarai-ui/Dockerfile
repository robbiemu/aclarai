# aclarai UI Service - Gradio Frontend
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

# Copy the entire project context into the image. This makes the root
# pyproject.toml and all shared/service code available to the installer.
# (Make sure you have a .dockerignore file at the root to exclude .git, etc.)
COPY . .

# Install all workspace dependencies using the root pyproject.toml, and the UI-specific dependencies from its own pyproject.toml
RUN uv pip install --system -e ./services/aclarai-ui

# Create a non-root user for security
RUN useradd -m -u 1000 aclarai && chown -R aclarai:aclarai /app
USER aclarai

# Expose the port that Gradio runs on (7860 is Gradio default)
EXPOSE 7860

# Health check (basic HTTP check on Gradio interface)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python", "-m", "aclarai_ui.main"]