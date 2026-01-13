# Installation

This guide covers installing Data Miner and its dependencies.

---

## Prerequisites

- **Python 3.12+**
- **PostgreSQL 15+** (local or remote)
- **FFmpeg** (for video processing)

---

## Quick Install

```bash
# Clone repository
git clone https://github.com/tycoai/data_miner.git
cd data_miner

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

---

## PostgreSQL Setup

### Option 1: Docker Compose (Recommended)

```bash
# Start PostgreSQL container
docker compose up -d

# Verify it's running
docker compose ps
```

The included `docker-compose.yaml` starts PostgreSQL on port 5432 with default credentials.

### Option 2: Local PostgreSQL

```bash
# Create database
createdb data_miner

# Or via psql
psql -c "CREATE DATABASE data_miner;"
```

---

## Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# Database URL (default works with Docker Compose)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/data_miner

# Hugging Face token (optional, for private models)
# Get from: https://huggingface.co/settings/tokens
HF_TOKEN=your_token_here

# Debug mode (disables heartbeat - for development only)
# DATA_MINER_DEBUG=1
```

---

## Initialize Database

```bash
# Create tables
data-miner init-db

# Verify connection
data-miner status
```

---

## GPU Setup (Optional)

For ML inference, ensure CUDA is available:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

The pipeline automatically falls back to CPU if CUDA is unavailable.

---

## Verify Installation

```bash
# Check CLI is available
data-miner --help

# Check database connection
data-miner status
```

---

## Next Steps

- [Configuration](configuration.md) - Set up your pipeline config
- [Quickstart](quickstart.md) - Run your first pipeline
