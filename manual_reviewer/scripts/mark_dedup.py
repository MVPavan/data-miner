"""Apply dedup survivor/cluster assignments to an aa_v4 ``pipeline.db``.

Two manifest formats supported:

1. **Cluster manifest** (``--manifest-clusters``): JSON array, one entry
   per cluster. Each entry names the survivor and the dropped frames::

       [
         {"cluster_id": "c001",
          "survivor": "frame_0001",
          "dropped":  ["frame_0002", "frame_0003"]},
         ...
       ]

   The cluster_id is written verbatim into ``image_meta.dedup_cluster_id``
   for every member (survivor + dropped). Survivors get
   ``dedup_status='survivor'``; the rest get ``'dropped'``.

2. **Flat manifest** (``--manifest-flat``): JSON array of records, one per
   image::

       [
         {"image_id": "frame_0001", "cluster_id": "c001", "is_survivor": true},
         {"image_id": "frame_0002", "cluster_id": "c001", "is_survivor": false},
         ...
       ]

   Easier to produce from a Pandas frame; equivalent in effect.

The script does not run the dedup model itself — that's the upstream
``data_miner/modules/deduplicator.py`` pipeline's job. This is the thin
adapter that lands its output in the DB so ``build_tasks`` filters
non-survivors out of review.

Usage::

    python -m manual_reviewer.scripts.mark_dedup \\
        --db /jobs/run_42/pipeline.db \\
        --manifest-clusters /tmp/dedup_clusters.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from manual_reviewer.pipeline_io import write_dedup_assignments

logger = logging.getLogger("manual_reviewer.mark_dedup")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.db.exists():
        logger.error("pipeline.db not found: %s", args.db)
        return 2

    if args.manifest_clusters:
        assignments = list(_walk_clusters(json.loads(args.manifest_clusters.read_text(encoding="utf-8"))))
    elif args.manifest_flat:
        assignments = list(_walk_flat(json.loads(args.manifest_flat.read_text(encoding="utf-8"))))
    else:
        logger.error("provide --manifest-clusters or --manifest-flat")
        return 2

    if not assignments:
        logger.warning("manifest produced 0 assignments")
        return 1

    survivors = sum(1 for _, _, is_s in assignments if is_s)
    drops = len(assignments) - survivors
    logger.info(
        "applying %d assignments to %s (survivors=%d, dropped=%d)",
        len(assignments), args.db, survivors, drops,
    )
    updated = write_dedup_assignments(args.db, assignments)
    logger.info("rows updated: %d", updated)
    if updated < len(assignments):
        logger.warning(
            "%d assignment(s) had no matching image_meta row — verify ids",
            len(assignments) - updated,
        )
    return 0


def _walk_clusters(raw: Any):
    if not isinstance(raw, list):
        raise ValueError("manifest-clusters must be a JSON array")
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        cluster_id = entry.get("cluster_id")
        survivor = entry.get("survivor")
        dropped = entry.get("dropped") or []
        if survivor:
            yield (str(survivor), str(cluster_id) if cluster_id is not None else None, True)
        for d in dropped:
            yield (str(d), str(cluster_id) if cluster_id is not None else None, False)


def _walk_flat(raw: Any):
    if not isinstance(raw, list):
        raise ValueError("manifest-flat must be a JSON array")
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        image_id = entry.get("image_id")
        if not image_id:
            continue
        cluster_id = entry.get("cluster_id")
        is_survivor = bool(entry.get("is_survivor", True))
        yield (str(image_id), str(cluster_id) if cluster_id is not None else None, is_survivor)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply dedup assignments to aa_v4 pipeline.db")
    p.add_argument("--db", type=Path, required=True)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--manifest-clusters", type=Path, default=None)
    src.add_argument("--manifest-flat", type=Path, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
