FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY data_miner/ ./data_miner/

# Install uv and dependencies
RUN pip install uv && \
    uv pip install --system --no-cache-dir -e .

# Create non-root user
RUN useradd -m -u 1000 dataminer
USER dataminer

# Health check file location
ENV HEALTH_FILE=/tmp/healthy

# Default command (overridden in K8s)
ENTRYPOINT ["python", "-m"]
CMD ["data_miner.workers.download"]
