"""Post-step: compact fragments and build ANN indexes on a LanceDB table.

Run once after all 4 shard processes finish. Compaction merges the many
small fragments that per-batch appends produce; IVF_PQ makes similarity
search fast.

Usage::

    python -m data_miner.embeddings.compact_and_index \\
        --lance-uri /mnt/data/data_miner_lance/laion_good2 \\
        --table embeddings \\
        [--columns siglip2_embedding dinov3_embedding] \\
        [--accelerator cuda] \\
        [--skip-index]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import lancedb
import pyarrow as pa

from .lance_store import LANCE_ROOT_DEFAULT, TABLE_DEFAULT

log = logging.getLogger("compact_and_index")


def _is_fixed_size_list(col_type: pa.DataType) -> bool:
    return pa.types.is_fixed_size_list(col_type)


def _nonnull_count(tbl, column: str) -> int:
    """Count rows where ``column`` is non-null using the native filter API."""
    if tbl.count_rows() == 0:
        return 0
    return tbl.count_rows(filter=f"{column} IS NOT NULL")


def _resolve_sub_vectors(dim: int, requested: int, min_sub_vectors: int = 4) -> int:
    """Snap ``requested`` down to the largest divisor of ``dim`` that is
    still >= ``min_sub_vectors``. Raises if no such divisor exists (which
    would imply a pathological dim).
    """
    if requested < min_sub_vectors:
        raise ValueError(f"num_sub_vectors {requested} < floor {min_sub_vectors}")
    for k in range(min(requested, dim), min_sub_vectors - 1, -1):
        if dim % k == 0:
            return k
    raise ValueError(
        f"no divisor of {dim} in [{min_sub_vectors}, {requested}]; "
        f"pick a different num-sub-vectors or embedding dim"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lance-uri", type=str,
                    default=f"{LANCE_ROOT_DEFAULT}/laion_good2")
    ap.add_argument("--table", type=str, default=TABLE_DEFAULT)
    ap.add_argument("--columns", nargs="*",
                    default=["siglip2_embedding", "dinov3_embedding"],
                    help="Vector columns to index (skipped if all-null)")
    ap.add_argument("--skip-compact", action="store_true")
    ap.add_argument("--skip-index", action="store_true")
    ap.add_argument("--metric", type=str, default="cosine",
                    choices=["cosine", "l2", "dot"])
    ap.add_argument("--num-partitions", type=int, default=2048,
                    help="IVF partitions; sqrt(N) ≈ 2300 for 5.4M so 2048 fits")
    ap.add_argument("--num-sub-vectors", type=int, default=48,
                    help="PQ sub-vectors; snapped down to a divisor of the "
                         "embedding dim if not divisible")
    ap.add_argument("--min-sub-vectors", type=int, default=4,
                    help="Floor for the divisor snap (fewer is degenerate)")
    ap.add_argument("--accelerator", type=str, default=None,
                    help="e.g. 'cuda' to train IVF centroids on GPU "
                         "(requires lance built with CUDA support); "
                         "default = CPU")
    ap.add_argument("--target-rows-per-fragment", type=int, default=1_000_000,
                    help="Target rows per fragment after compaction. Default "
                         "1M keeps the post-compact fragment count small "
                         "(~5-6 fragments for a 5.4M-row table) without "
                         "blowing out memory during compact.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    uri = str(Path(args.lance_uri))
    db = lancedb.connect(uri)
    if args.table not in db.table_names():
        log.error("table not found: %s/%s", uri, args.table)
        return 2
    tbl = db.open_table(args.table)
    log.info("table %s/%s: %d rows", uri, args.table, tbl.count_rows())

    if not args.skip_compact:
        log.info("compacting fragments (target_rows_per_fragment=%d) ...",
                 args.target_rows_per_fragment)
        t0 = time.perf_counter()
        try:
            stats = tbl.compact_files(
                target_rows_per_fragment=args.target_rows_per_fragment,
            )
        except TypeError as e:
            # Older lancedb versions may not forward the kwarg.
            log.warning("compact_files kwarg unsupported (%s); retrying "
                        "with defaults", e)
            stats = tbl.compact_files()
        # stats is a small dataclass; log its attrs explicitly rather than
        # relying on repr staying stable across lance versions.
        attrs = {k: getattr(stats, k) for k in
                 ("fragments_removed", "fragments_added",
                  "files_removed", "files_added")
                 if hasattr(stats, k)}
        log.info("compaction done in %.1fs | %s",
                 time.perf_counter() - t0, attrs or stats)

    if args.skip_index:
        log.info("skip-index set; done")
        return 0

    schema = tbl.schema
    for col in args.columns:
        if col not in schema.names:
            log.info("column %s not in schema, skip", col)
            continue
        field = schema.field(col)
        if not _is_fixed_size_list(field.type):
            log.info("column %s is not a vector column, skip", col)
            continue
        nn = _nonnull_count(tbl, col)
        if nn == 0:
            log.info("column %s has 0 non-null rows, skip index", col)
            continue

        dim = field.type.list_size
        try:
            nsv = _resolve_sub_vectors(
                dim, args.num_sub_vectors, args.min_sub_vectors,
            )
        except ValueError as e:
            log.error("column %s: %s; skipping this column", col, e)
            continue
        if nsv != args.num_sub_vectors:
            log.warning(
                "column %s: num_sub_vectors %d doesn't divide dim %d; "
                "using %d (sub-vector size = %d)",
                col, args.num_sub_vectors, dim, nsv, dim // nsv,
            )

        log.info("creating IVF_PQ index on %s (%d non-null, dim=%d, "
                 "partitions=%d, sub_vectors=%d, metric=%s%s) ...",
                 col, nn, dim, args.num_partitions, nsv, args.metric,
                 f", accelerator={args.accelerator}" if args.accelerator else "")
        t0 = time.perf_counter()
        kwargs = dict(
            vector_column_name=col,
            num_partitions=args.num_partitions,
            num_sub_vectors=nsv,
            metric=args.metric,
            index_type="IVF_PQ",
            replace=True,
        )
        if args.accelerator:
            kwargs["accelerator"] = args.accelerator
        tbl.create_index(**kwargs)
        log.info("index on %s built in %.1fs",
                 col, time.perf_counter() - t0)

    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
