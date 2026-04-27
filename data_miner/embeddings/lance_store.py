"""LanceDB schema + writer for multi-model image embeddings.

One table per dataset. Columns for multiple embedding models live side
by side (SigLIP2 now; DINOv3 added later with ``tbl.add_columns`` or
filled in-place, no rewrite of SigLIP2 bytes).
"""

from __future__ import annotations

import atexit
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa

from ..logging import get_logger

logger = get_logger(__name__)


# ---- dims (kept in one place; update if you swap model variants) -----------
#
# SigLIP2 image feature dim == vision_config.hidden_size (no projection head
# in SigLIP2). Verified from transformers AutoConfig:
#   so400m-patch14-384    -> 1152
#   giant-opt-patch16-384 -> 1536
#
# DINOv3 feature dim == hidden_size:
#   dinov3-small 384, base 768, large 1024, huge 1280, giant 1536.

SIGLIP2_DIMS: dict[str, int] = {
    "siglip2-so400m": 1152,
    "siglip2-giant":  1536,
}
SIGLIP2_DIM = 1536   # default = giant (the active choice)

DINOV3_DIMS: dict[str, int] = {
    "dinov2-base":  768,
    "dinov2-large": 1024,
    "dinov3-small": 384,
    "dinov3-base":  768,
    "dinov3-large": 1024,
    "dinov3-huge":  1280,
    "dinov3-giant": 1536,
}
DINOV3_DIM = 768     # default = dinov3-base

LANCE_ROOT_DEFAULT = "/mnt/data/data_miner_lance"
TABLE_DEFAULT = "embeddings"
ERRORS_TABLE_DEFAULT = "embed_errors"

# Prefer the typed CommitConflictError; fall back to a narrow string-match
# for older lance builds. Do NOT include loose tokens like "concurrent" or
# "conflicting" — those also match schema and arrow errors and would cause
# fatal errors to be retried 8 times with backoff instead of surfaced.
try:  # pragma: no cover
    from lance.commit import CommitConflictError as _LanceCommitConflictError
except Exception:  # pragma: no cover
    _LanceCommitConflictError = None  # type: ignore[assignment]

_COMMIT_CONFLICT_SIGNATURES = (
    "commit conflict",
    "commitconflict",
)


# ---- schemas ---------------------------------------------------------------


def build_schema(
    siglip2_dim: int = SIGLIP2_DIM,
    dinov3_dim: int = DINOV3_DIM,
) -> pa.Schema:
    """Main embeddings table. fixed-size lists so LanceDB can index them."""
    return pa.schema([
        pa.field("image_id",  pa.string(), nullable=False),
        pa.field("image_path", pa.string(), nullable=False),
        pa.field("subset",     pa.string(), nullable=False),
        pa.field("siglip2_embedding",
                 pa.list_(pa.float16(), siglip2_dim), nullable=True),
        pa.field("dinov3_embedding",
                 pa.list_(pa.float16(), dinov3_dim),  nullable=True),
        pa.field("siglip2_model_id", pa.string(), nullable=True),
        pa.field("dinov3_model_id",  pa.string(), nullable=True),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ])


def build_errors_schema() -> pa.Schema:
    """Sidecar table for decode/inference failures (so resume skips them)."""
    return pa.schema([
        pa.field("image_id",  pa.string(), nullable=False),
        pa.field("image_path", pa.string(), nullable=False),
        pa.field("subset",     pa.string(), nullable=False),
        pa.field("stage",      pa.string(), nullable=False),   # "decode" | "infer"
        pa.field("error",      pa.string(), nullable=False),
        pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
    ])


# ---- row dict -> pa.Table coercion ----------------------------------------


def _vec_to_list(v, dim: int) -> Optional[list]:
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float16).reshape(-1)
    if arr.size != dim:
        raise ValueError(f"embedding length {arr.size} != schema dim {dim}")
    return arr.tolist()


def rows_to_arrow(rows: list[dict], schema: pa.Schema) -> pa.Table:
    """Coerce a list of plain dicts into a pa.Table matching ``schema``.

    Numpy float16 arrays in ``*_embedding`` fields are converted to Python
    lists, which pyarrow then packs into the fixed_size_list per the schema.
    ``None`` is preserved as a null row.
    """
    if not rows:
        return schema.empty_table()

    dims = {f.name: f.type.list_size for f in schema
            if pa.types.is_fixed_size_list(f.type)}

    cooked = []
    for r in rows:
        out = dict(r)
        for col, dim in dims.items():
            out[col] = _vec_to_list(out.get(col), dim)
        cooked.append(out)

    df = pd.DataFrame(cooked, columns=schema.names)
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False)


# ---- writer ---------------------------------------------------------------


def _is_commit_conflict(exc: BaseException) -> bool:
    if _LanceCommitConflictError is not None and isinstance(exc, _LanceCommitConflictError):
        return True
    s = str(exc).lower()
    return any(tok in s for tok in _COMMIT_CONFLICT_SIGNATURES)


class LanceEmbeddingWriter:
    """Thin, buffered, crash-safe append writer for one LanceDB table.

    Multiple processes may hold writers against the same table — LanceDB
    uses optimistic manifest commits; each append creates a new fragment.
    This class adds its own retry-with-backoff on the (rare) commit
    conflict error so a lost race doesn't drop a whole buffer.
    """

    def __init__(
        self,
        uri: str | Path,
        table: str,
        schema: pa.Schema,
        buffer_rows: int = 4096,
        create_if_missing: bool = True,
        max_commit_retries: int = 8,
    ) -> None:
        import lancedb  # local import so module imports cheaply
        uri = str(Path(uri))
        Path(uri).mkdir(parents=True, exist_ok=True)

        self.uri = uri
        self.table_name = table
        self.schema = schema
        self.buffer_rows = buffer_rows
        self.max_commit_retries = max_commit_retries
        self._buf: list[dict] = []
        self._lock = RLock()
        self._closed = False

        self._db = lancedb.connect(uri)
        existing = set(self._db.table_names())
        if table in existing:
            self._tbl = self._db.open_table(table)
            logger.info("opened existing table %s/%s (%d rows)",
                        uri, table, self._tbl.count_rows())
        elif create_if_missing:
            self._tbl = self._db.create_table(table, schema=schema)
            logger.info("created table %s/%s", uri, table)
        else:
            raise FileNotFoundError(f"table not found: {uri}/{table}")

        atexit.register(self._atexit_flush)

    # ---- public API ------------------------------------------------------

    @property
    def table(self):
        return self._tbl

    def add(self, row: dict) -> None:
        """Buffer one row; flush when buffer fills."""
        with self._lock:
            self._buf.append(row)
            if len(self._buf) >= self.buffer_rows:
                self._flush_locked()

    def extend(self, rows: list[dict]) -> None:
        """Buffer many rows; may trigger multiple flushes."""
        with self._lock:
            self._buf.extend(rows)
            while len(self._buf) >= self.buffer_rows:
                chunk, self._buf = (
                    self._buf[: self.buffer_rows],
                    self._buf[self.buffer_rows :],
                )
                self._write_arrow(rows_to_arrow(chunk, self.schema))

    def flush(self) -> int:
        with self._lock:
            return self._flush_locked()

    def close(self) -> int:
        """Flush buffer and mark writer closed.

        Always marks ``_closed = True`` even if flush raises, so the atexit
        handler won't re-attempt the same failing flush (which masks the
        original traceback and can wedge shutdown). Use a fresh writer if
        you want to retry after fixing the underlying error.
        """
        with self._lock:
            try:
                return self._flush_locked()
            finally:
                self._closed = True

    def count_rows(self) -> int:
        return self._tbl.count_rows()

    def existing_image_ids(self, shard_filter: Optional[set[str]] = None) -> set[str]:
        """Return the set of ``image_id`` values currently in the table.

        If ``shard_filter`` is given, results are the intersection with it
        and the scan streams — never materializing the full id universe.
        Use this from each shard to keep resume-startup memory per-process
        bounded to ``len(shard_filter)`` instead of the whole table.

        Prefers the pylance streaming scanner; only falls back to the
        search()-based path if pylance is unavailable. Transient scanner
        errors propagate (they can indicate a corrupted fragment that
        silently truncating the scan would turn into a duplicate-rows bug
        on resume).
        """
        n = self._tbl.count_rows()
        if n == 0:
            return set()

        # Preferred: pylance streaming scanner
        try:
            ds = self._tbl.to_lance()
        except (ImportError, AttributeError) as e:
            logger.info("pylance scanner unavailable (%s); using search() fallback", e)
            ds = None

        if ds is not None:
            scanner = ds.scanner(columns=["image_id"])
            reader = scanner.to_reader()  # RecordBatchReader — streams
            out: set[str] = set()
            if shard_filter is not None:
                # Intersect per-batch; do not build the full id set.
                for rb in reader:
                    for i in rb.column("image_id").to_pylist():
                        if i in shard_filter:
                            out.add(i)
            else:
                for rb in reader:
                    out.update(rb.column("image_id").to_pylist())
            return out

        # Fallback: this reads the full id column in one shot. Prefer
        # installing pylance so the streaming path is used instead.
        t = self._tbl.search().select(["image_id"]).limit(n).to_arrow()
        ids = t.column("image_id").to_pylist()
        if shard_filter is not None:
            return {i for i in ids if i in shard_filter}
        return set(ids)

    # ---- internals -------------------------------------------------------

    def _write_arrow(self, tbl: pa.Table) -> None:
        """Append an Arrow table; retry on commit conflicts with backoff."""
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_commit_retries + 1):
            try:
                self._tbl.add(tbl)
                if attempt > 0:
                    logger.info("lance commit succeeded on retry %d", attempt)
                return
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if not _is_commit_conflict(e) or attempt == self.max_commit_retries:
                    raise
                delay = 0.5 * (2 ** attempt) + random.uniform(0, 0.25)
                logger.warning(
                    "commit conflict (attempt %d/%d): %s — sleeping %.2fs",
                    attempt + 1, self.max_commit_retries, e, delay,
                )
                time.sleep(delay)
        if last_exc is not None:  # unreachable defensive
            raise last_exc

    def _flush_locked(self) -> int:
        if not self._buf:
            return 0
        n = len(self._buf)
        self._write_arrow(rows_to_arrow(self._buf, self.schema))
        self._buf.clear()
        return n

    def _atexit_flush(self) -> None:
        try:
            if self._closed:
                return
            n = self.flush()
            if n:
                logger.info("atexit flush: %d rows written to %s/%s",
                            n, self.uri, self.table_name)
        except Exception:
            logger.exception("atexit flush for %s/%s failed",
                             self.uri, self.table_name)


# ---- helpers --------------------------------------------------------------


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
