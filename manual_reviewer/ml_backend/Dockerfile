# Manual reviewer LS ML backend — protocol adapter only.
#
# Talks HTTP to the SAM 3.1 LitServe server (port 3014 on the host network)
# and reads pipeline.db over a bind mount. No model weights live in this
# container, so the image stays small and CPU-only.
#
# Build:
#   docker build -f manual_reviewer/ml_backend/Dockerfile -t mr-ml-backend .
#
# Run inside docker-compose.review.yml; see that file for required env vars.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# label-studio-ml provides the LabelStudioMLBase + Flask harness. Pinned
# loosely; the LS version on the LS frontend container governs wire format.
RUN pip install \
    "label-studio-ml>=1.0.9" \
    "requests>=2.31" \
    "pydantic>=2.0"

COPY data_miner/auto_annotation_v4/configs /app/data_miner/auto_annotation_v4/configs
COPY manual_reviewer/__init__.py /app/manual_reviewer/__init__.py
COPY manual_reviewer/reconcile /app/manual_reviewer/reconcile
COPY manual_reviewer/ml_backend /app/manual_reviewer/ml_backend

# Stub package __init__ files so the imports above resolve. The aav4 configs
# subtree pulls in only the wire / contracts / enums modules — no torch.
RUN python -c "import pathlib; pathlib.Path('/app/data_miner/__init__.py').touch(); pathlib.Path('/app/data_miner/auto_annotation_v4/__init__.py').touch()"

ENV PYTHONPATH=/app

EXPOSE 9090

CMD ["python", "-m", "manual_reviewer.ml_backend.server"]
