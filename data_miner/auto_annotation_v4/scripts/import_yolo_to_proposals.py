"""Import YOLO .txt labels into a pipeline.db as detect-stage proposals.

Inverse of ``export_yolo_from_proposals.py``. Reads a directory of YOLO-format
``.txt`` files (``class_id cx cy w h [score]``, one per image), resolves the
corresponding image path from an image directory, and seeds a pipeline.db
with the rows filter/evaluate/finalize expect:

  * ``image_meta``      — one row per image (status + stages_completed=['detect'])
  * ``proposals``       — one row per image keyed by model (default: sam3_dart)
  * ``stages``          — one row per image under stage='detect' (merged view)
  * ``work_queue``      — one row per image with stage='filter', pending

Use this when detect was run in standalone mode (producing ``labels_standalone/``)
and you want filter/evaluate/finalize to operate on those labels without
re-running GPU inference.

Usage:
    python -m data_miner.auto_annotation_v4.scripts.import_yolo_to_proposals \
        <JOB_DIR> --image-dir /path/to/images \
        [--labels-dir JOB_DIR/labels_standalone] \
        [--model sam3_dart] \
        [--min-score 0.0]

The ``<JOB_DIR>`` must contain a ``config.yaml`` with a ``class_registry`` so
class ids can be mapped back to canonical names. If no ``pipeline.db`` exists
in ``<JOB_DIR>``, one is created.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

log = logging.getLogger("import_yolo_to_proposals")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Schema (mirrors checkpoint.py _SCHEMA — inline so the importer can stand
# alone and create a fresh DB from scratch without going through the async
# aiosqlite path)
# ---------------------------------------------------------------------------

_SCHEMA = """\
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS job_info (
    job_id          TEXT NOT NULL,
    image_dir       TEXT,
    config_hash     TEXT NOT NULL,
    prompt_version  TEXT NOT NULL,
    created_at      REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS image_meta (
    image_id         TEXT PRIMARY KEY,
    image_path       TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    stages_completed TEXT NOT NULL DEFAULT '[]',
    config_hash      TEXT NOT NULL DEFAULT '',
    prompt_version   TEXT NOT NULL DEFAULT '',
    total_timing_ms  REAL NOT NULL DEFAULT 0.0,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS proposals (
    image_id    TEXT NOT NULL,
    model       TEXT NOT NULL,
    data        TEXT NOT NULL,
    config_hash TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    PRIMARY KEY (image_id, model)
);
CREATE INDEX IF NOT EXISTS idx_proposals_model ON proposals(model);

CREATE TABLE IF NOT EXISTS stages (
    image_id    TEXT NOT NULL,
    stage       TEXT NOT NULL,
    data        TEXT NOT NULL,
    config_hash TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    PRIMARY KEY (image_id, stage)
);
CREATE INDEX IF NOT EXISTS idx_stages_stage ON stages(stage);

CREATE TABLE IF NOT EXISTS work_queue (
    image_id    TEXT NOT NULL,
    stage       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    worker_id   TEXT,
    score       REAL NOT NULL,
    claimed_at  REAL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (image_id, stage)
);
CREATE INDEX IF NOT EXISTS idx_wq_claim ON work_queue(stage, status, score);

CREATE TABLE IF NOT EXISTS failures (
    image_id        TEXT NOT NULL,
    stage           TEXT NOT NULL,
    attempts        INTEGER NOT NULL DEFAULT 1,
    last_error      TEXT,
    last_attempt_at REAL,
    PRIMARY KEY (image_id, stage)
);
"""


# ---------------------------------------------------------------------------
# Class id <-> name map (mirrors the exporter)
# ---------------------------------------------------------------------------


def load_id_to_class(job_dir: Path) -> dict[int, str]:
    """Load {class_id: class_name} from config.yaml's class_registry.

    The exporter's forward direction is class_name → class_id; this inverts
    to go back. Falls back to positional classes.txt only if config.yaml is
    absent.
    """
    cfg_path = job_dir / "config.yaml"
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            data = None
        if isinstance(data, dict):
            reg = data.get("class_registry") or {}
            if isinstance(reg, dict):
                out: dict[int, str] = {}
                for name, cls in reg.items():
                    if isinstance(cls, dict) and "id" in cls:
                        out[int(cls["id"])] = name
                if out:
                    return out

    classes_txt = job_dir / "classes.txt"
    if classes_txt.exists():
        names = [
            ln.strip() for ln in classes_txt.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        return {i: n for i, n in enumerate(names)}

    return {}


# ---------------------------------------------------------------------------
# YOLO parsing
# ---------------------------------------------------------------------------


def parse_yolo_line(line: str) -> tuple[int, float, float, float, float, float] | None:
    """Parse one YOLO row. Returns (class_id, cx, cy, w, h, score) or None."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cid = int(parts[0])
        cx, cy, w, h = (float(p) for p in parts[1:5])
        score = float(parts[5]) if len(parts) >= 6 else 1.0
    except (ValueError, TypeError):
        return None
    return cid, cx, cy, w, h, score


def yolo_to_bbox(
    cx: float, cy: float, w: float, h: float
) -> tuple[float, float, float, float]:
    """(cx, cy, w, h) normalized → (x1, y1, x2, y2), clamped to [0,1]."""
    x1 = max(0.0, min(1.0, cx - w / 2.0))
    y1 = max(0.0, min(1.0, cy - h / 2.0))
    x2 = max(0.0, min(1.0, cx + w / 2.0))
    y2 = max(0.0, min(1.0, cy + h / 2.0))
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------


def build_image_path_index(image_dir: Path) -> dict[str, str]:
    """Scan *image_dir* and return {stem: absolute_path} for every image.

    Works only at the top level (no recursion). That matches how the
    standalone runner was invoked for fl_pj.
    """
    out: dict[str, str] = {}
    for p in image_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        out[p.stem] = str(p.resolve())
    return out


def read_image_size(path: str) -> tuple[int, int]:
    """PIL header-only read. Returns (w, h) or (0, 0) on failure."""
    try:
        with Image.open(path) as im:
            return im.size
    except Exception:
        return (0, 0)


# ---------------------------------------------------------------------------
# Build proposal + detect result payload for one image
# ---------------------------------------------------------------------------


def build_payloads(
    *,
    image_id: str,
    image_path: str,
    image_w: int,
    image_h: int,
    candidates_payload: list[dict],
    model_name: str,
) -> tuple[str, str]:
    """Returns (proposal_json, detect_json) pre-serialised for sqlite insert.

    Both payloads reference the same candidate list. The proposal payload
    is the per-model view; the detect-stage payload is the merged view
    (here identical because there's only one detector).
    """
    proposal_payload = {
        "model": model_name,
        "image_id": image_id,
        "image_size": [image_w, image_h],
        "latency_ms": 0.0,
        "candidates": candidates_payload,
    }
    detect_payload = {
        "image_id": image_id,
        "image_path": image_path,
        "image_size": [image_w, image_h],
        "models_used": [model_name],
        "candidates": candidates_payload,
        "routing": {
            "auto_accepted": [],
            "needs_evaluation": [],
            "confusion_flags": [],
        },
        "filter_stats": {},
        "stage_timing_ms": 0.0,
    }
    return json.dumps(proposal_payload), json.dumps(detect_payload)


def build_candidate(
    *,
    image_id: str,
    class_name: str,
    bbox: tuple[float, float, float, float],
    score: float,
    source_model: str,
    idx: int,
) -> dict:
    x1, y1, x2, y2 = bbox
    return {
        "candidate_id": f"{image_id}_{source_model}_{idx}_{uuid.uuid4().hex[:8]}",
        "class_name": class_name,
        "label": class_name,
        "source_model": source_model,
        "expression": class_name,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "score": float(score),
        "agreement": 1,
        "agreeing_models": [source_model],
        "status": "proposed",
        "mask_rle": None,
        "metadata": {"imported_from": "yolo_standalone"},
        "notes": [],
    }


# ---------------------------------------------------------------------------
# Main ingestion loop
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job_dir", type=Path, help="Pipeline job directory")
    ap.add_argument(
        "--image-dir", type=Path, required=True,
        help="Root dir containing the source images (flat, no recursion)",
    )
    ap.add_argument(
        "--labels-dir", type=Path, default=None,
        help="YOLO .txt dir (default: <job_dir>/labels_standalone)",
    )
    ap.add_argument(
        "--model", default="sam3_dart",
        help="source_model value stamped onto each Candidate (default: sam3_dart)",
    )
    ap.add_argument(
        "--min-score", type=float, default=0.0,
        help="Drop YOLO rows below this score (default: 0 = keep all)",
    )
    ap.add_argument(
        "--job-id", default=None,
        help="Override job_info.job_id (default: <job_dir>.name)",
    )
    ap.add_argument(
        "--image-size-workers", type=int, default=16,
        help="Threads for header-only image-size reads (default: 16)",
    )
    ap.add_argument(
        "--skip-image-size", action="store_true",
        help="Skip per-image header reads; use (0, 0) placeholders. Faster; "
             "downstream filter/evaluate reads actual dims from the image "
             "file at runtime anyway.",
    )
    ap.add_argument(
        "--batch-size", type=int, default=500,
        help="SQL insert batch size (default: 500)",
    )
    ap.add_argument(
        "--config", type=Path, default=None,
        help="Override YAML whose merged config_hash should be stamped onto "
             "the imported rows. Required for the pipeline to pick up the "
             "imported detect checkpoint without invalidating it via "
             "config-hash drift. If omitted, rows are stamped with an empty "
             "hash and the pipeline will clear_downstream them on first run.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    job_dir: Path = args.job_dir.resolve()
    image_dir: Path = args.image_dir.resolve()
    labels_dir: Path = (args.labels_dir or (job_dir / "labels_standalone")).resolve()
    db_path = job_dir / "pipeline.db"

    if not job_dir.exists():
        log.error("job dir does not exist: %s", job_dir)
        return 2
    if not image_dir.exists():
        log.error("image dir does not exist: %s", image_dir)
        return 2
    if not labels_dir.exists():
        log.error("labels dir does not exist: %s", labels_dir)
        return 2

    id_to_class = load_id_to_class(job_dir)
    if not id_to_class:
        log.error("no class mapping (config.yaml or classes.txt) in %s", job_dir)
        return 3
    log.info("loaded %d classes from %s", len(id_to_class), job_dir)

    # Compute the pipeline's config_hash so the submitter doesn't treat our
    # imported detect rows as stale and clear_downstream them.
    stamped_config_hash = ""
    if args.config:
        from data_miner.auto_annotation_v4.configs.loader import (
            compute_config_hash, load_config,
        )
        try:
            cfg = load_config(str(args.config))
            stamped_config_hash = compute_config_hash(cfg, cfg.prompts_dir)
            log.info("stamping config_hash=%s from %s",
                     stamped_config_hash[:12] + "...", args.config)
        except Exception as e:
            log.error("failed to compute config_hash from %s: %s", args.config, e)
            return 4
    else:
        log.warning(
            "no --config given; imported rows will have empty config_hash "
            "and the pipeline will clear_downstream them on first run"
        )

    log.info("indexing images under %s ...", image_dir)
    t0 = time.time()
    img_index = build_image_path_index(image_dir)
    log.info("  indexed %d images in %.1fs", len(img_index), time.time() - t0)

    log.info("scanning labels %s ...", labels_dir)
    txt_files = sorted(labels_dir.glob("*.txt"))
    log.info("  found %d .txt files", len(txt_files))

    # Pre-resolve image paths + (optionally) image sizes for all images
    # that have a .txt. Done up front so the main loop is a tight writer.
    items: list[tuple[str, str, int, int, Path]] = []
    missing_images = 0
    for txt in txt_files:
        image_id = txt.stem
        image_path = img_index.get(image_id)
        if image_path is None:
            missing_images += 1
            continue
        items.append((image_id, image_path, 0, 0, txt))
    if missing_images:
        log.warning(
            "%d .txt files had no matching image under %s — skipped",
            missing_images, image_dir,
        )
    log.info("will import %d (image, labels) pairs", len(items))

    if not args.skip_image_size and items:
        log.info("reading %d image headers with %d threads ...",
                 len(items), args.image_size_workers)
        t0 = time.time()
        paths_by_idx = {i: it[1] for i, it in enumerate(items)}
        sizes: dict[int, tuple[int, int]] = {}
        with ThreadPoolExecutor(max_workers=args.image_size_workers) as ex:
            futures = {
                ex.submit(read_image_size, p): i for i, p in paths_by_idx.items()
            }
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                sizes[i] = fut.result()
                done += 1
                if done % 5000 == 0:
                    log.info("  %d/%d headers read", done, len(items))
        items = [
            (iid, ipath, sizes[i][0], sizes[i][1], txt)
            for i, (iid, ipath, _, _, txt) in enumerate(items)
        ]
        log.info("  headers read in %.1fs", time.time() - t0)

    # ---- connect + create schema ----
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
    try:
        con.executescript(_SCHEMA)

        # Register job_info if empty (idempotent: only set on first run).
        row = con.execute("SELECT COUNT(*) FROM job_info").fetchone()
        if row[0] == 0:
            job_id = args.job_id or job_dir.name
            con.execute(
                "INSERT INTO job_info (job_id, image_dir, config_hash, prompt_version, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (job_id, str(image_dir), "", "", time.time()),
            )
            log.info("registered job_info for job_id=%s", job_id)

        now = time.time()
        total_detections = 0
        n_empty = 0
        skipped_class = 0
        skipped_bbox = 0
        skipped_score = 0
        unknown_class_hist: dict[int, int] = {}

        # ---- insert in batches ----
        img_meta_batch: list[tuple] = []
        proposals_batch: list[tuple] = []
        stages_batch: list[tuple] = []
        work_queue_batch: list[tuple] = []

        def flush() -> None:
            if not img_meta_batch:
                return
            con.execute("BEGIN")
            con.executemany(
                "INSERT OR REPLACE INTO image_meta"
                " (image_id, image_path, status, stages_completed,"
                "  config_hash, prompt_version, total_timing_ms,"
                "  created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                img_meta_batch,
            )
            con.executemany(
                "INSERT OR REPLACE INTO proposals"
                " (image_id, model, data, config_hash, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                proposals_batch,
            )
            con.executemany(
                "INSERT OR REPLACE INTO stages"
                " (image_id, stage, data, config_hash, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                stages_batch,
            )
            con.executemany(
                "INSERT OR IGNORE INTO work_queue"
                " (image_id, stage, status, score)"
                " VALUES (?, ?, ?, ?)",
                work_queue_batch,
            )
            con.execute("COMMIT")
            img_meta_batch.clear()
            proposals_batch.clear()
            stages_batch.clear()
            work_queue_batch.clear()

        for i, (image_id, image_path, image_w, image_h, txt) in enumerate(items, 1):
            # Parse YOLO lines
            try:
                raw = txt.read_text(encoding="utf-8")
            except OSError:
                continue
            candidates_payload: list[dict] = []
            for idx, line in enumerate(raw.splitlines()):
                parsed = parse_yolo_line(line)
                if parsed is None:
                    continue
                cid, cx, cy, w, h, score = parsed
                if score < args.min_score:
                    skipped_score += 1
                    continue
                class_name = id_to_class.get(cid)
                if class_name is None:
                    skipped_class += 1
                    unknown_class_hist[cid] = unknown_class_hist.get(cid, 0) + 1
                    continue
                bbox = yolo_to_bbox(cx, cy, w, h)
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    skipped_bbox += 1
                    continue
                candidates_payload.append(build_candidate(
                    image_id=image_id,
                    class_name=class_name,
                    bbox=bbox,
                    score=score,
                    source_model=args.model,
                    idx=idx,
                ))

            if not candidates_payload:
                n_empty += 1
            total_detections += len(candidates_payload)

            proposal_json, detect_json = build_payloads(
                image_id=image_id,
                image_path=image_path,
                image_w=image_w,
                image_h=image_h,
                candidates_payload=candidates_payload,
                model_name=args.model,
            )

            img_meta_batch.append((
                image_id, image_path, "running",
                json.dumps(["detect"]), stamped_config_hash, "", 0.0, now, now,
            ))
            proposals_batch.append((
                image_id, args.model, proposal_json, stamped_config_hash, now,
            ))
            stages_batch.append((
                image_id, "detect", detect_json, stamped_config_hash, now,
            ))
            work_queue_batch.append((
                image_id, "filter", "pending", now,
            ))

            if i % args.batch_size == 0:
                flush()
                if i % 5000 == 0:
                    log.info("  %d/%d imported", i, len(items))

        flush()

        log.info("")
        log.info("=== done ===")
        log.info("images imported:         %d", len(items))
        log.info("  with detections:       %d", len(items) - n_empty)
        log.info("  empty (no detections): %d", n_empty)
        log.info("total detections:        %d", total_detections)
        log.info("skipped bad bbox:        %d", skipped_bbox)
        log.info("skipped below min_score: %d (threshold=%.3f)",
                 skipped_score, args.min_score)
        log.info("skipped unknown class:   %d", skipped_class)
        if unknown_class_hist:
            top = sorted(unknown_class_hist.items(), key=lambda x: -x[1])[:10]
            log.info("  top unknown class ids: %s",
                     ", ".join(f"{cid}={n}" for cid, n in top))
        if missing_images:
            log.warning("(skipped %d .txt with no matching image)", missing_images)
    finally:
        con.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
