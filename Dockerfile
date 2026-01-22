# Data Miner - Docker Image
# Runs data-miner workers via supervisor

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    supervisor \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY data_miner/ ./data_miner/

# Install Python dependencies
RUN pip install --no-cache-dir uv && \
    uv pip install --system -e .

# Create directories for supervisor and logs
RUN mkdir -p /var/log/supervisor /etc/supervisor/conf.d

# Copy default config (will be overridden by volume mount)
COPY run_configs/run.yaml /app/config.yaml

# Environment variables
ENV DATA_MINER_CONFIG=/app/config.yaml
ENV PYTHONUNBUFFERED=1

# Default command - setup supervisor config and start
# The config.yaml determines which workers run
CMD ["sh", "-c", "data-miner workers setup --config $DATA_MINER_CONFIG && supervisord -n -c /etc/supervisor/supervisord.conf"]
