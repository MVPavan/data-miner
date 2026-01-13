# Contributing

Guidelines for contributing to Data Miner.

---

## Development Setup

```bash
# Clone and install in dev mode
git clone https://github.com/tycoai/data_miner.git
cd data_miner
pip install -e ".[dev]"

# Start PostgreSQL
docker compose up -d

# Initialize database
data-miner init-db
```

---

## Project Structure

```
data_miner/
├── cli.py              # Click CLI commands
├── config/             # Pydantic configs + OmegaConf
├── db/                 # SQLModel + PostgreSQL
├── workers/            # Long-running workers
├── modules/            # Core processing logic
├── models/             # ML model wrappers
└── utils/              # Utilities
```

---

## Code Style

- **Python 3.12+** features allowed
- **Type hints** required for public functions
- **Docstrings** for classes and public methods
- **Black** for formatting (line length 100)
- **isort** for imports

---

## Adding a New Worker

1. Create `workers/my_worker.py`
2. Extend appropriate base class:
   - `BaseVideoWorker` for per-video processing
   - `BaseProjectVideosWorker` for per-project-video processing
   - `BaseProjectStageWorker` for project-level operations

3. Implement `process()` method
4. Add to supervisor config in `cli.py`

---

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=data_miner
```

---

## Database Migrations

Currently using SQLModel's `create_all()`. For schema changes:

1. Update models in `db/models.py`
2. Run `data-miner init-db --force` (destroys data!)

---

## Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

---

## Reporting Issues

Include:
- Python version
- PostgreSQL version
- Config file (anonymized)
- Error logs
- Steps to reproduce
