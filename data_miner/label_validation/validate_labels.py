"""Validate YOLO labels against per-class SigLIP2 prompts.

For every ``(image_id, class_name)`` pair where the image's YOLO label file
lists a bbox of ``class_name``, score the whole image against that class's
positive / negative / junk prompts (threshold rule from
``data_miner.modules.frame_filter``) and write a verdict row to a sidecar
LanceDB table.

Design choices (see plan file for context):
- Whole-image scoring. If an image has N bboxes of the same class, we score
  that ``(image_id, class)`` pair once.
- Reuses the image embeddings already written to the LanceDB table; we never
  re-encode images. Only the text encoder runs on the GPU.
- Single streaming pass over the Lance table → per-batch matmul → verdict.
  End-to-end runtime on Good3 (~2 M rows, 1-2 classes) is disk-dominated.

Run::

    python -m data_miner.label_validation.validate_labels \\
        --config configs/label_validation/good3.yaml \\
        --lance-uri /mnt/data/data_miner_lance/laion_good3 \\
        --labels-dir output/auto_annotation_v4/laion_good3/labels_standalone \\
        --classes-yaml output/auto_annotation_v4/laion_good3/data.yaml \\
        --out-table label_validation \\
        --report-jsonl output/label_validation/good3_report.jsonl
"""

from __future__ import annotations

import argparse
import heapq
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import yaml

from ..config import SIGLIP2_MODELS
from ..embeddings.lance_store import LanceEmbeddingWriter
from ..models.siglip_model import SigLIPModel
from .config import ClassPrompts, ValidationConfig, load_config

log = logging.getLogger("validate_labels")


# ---- sidecar verdict table schema -----------------------------------------


def build_verdict_schema() -> pa.Schema:
    return pa.schema([
        pa.field("image_id",       pa.string(),  nullable=False),
        pa.field("class_name",     pa.string(),  nullable=False),
        pa.field("verdict",        pa.string(),  nullable=False),   # "pass" | "fail"
        pa.field("fail_reason",    pa.string(),  nullable=True),
        pa.field("pos_score_max",  pa.float32(), nullable=False),
        pa.field("neg_score_max",  pa.float32(), nullable=False),
        pa.field("junk_score_max", pa.float32(), nullable=False),
        pa.field("best_positive",  pa.string(),  nullable=True),
        pa.field("created_at",     pa.timestamp("us", tz="UTC"), nullable=False),
    ])


# ---- classes.txt / data.yaml loader ---------------------------------------


def load_id_to_name(path: Path) -> dict[int, str]:
    """Read a YOLO-style class map from either a ``data.yaml`` (``names:``
    block) or the sparse ``classes.txt`` our auto-annotation pipeline writes.

    Keys are class ids; values are the short class names used throughout
    the validation config.
    """
    text = path.read_text()
    # Prefer YAML if it parses and has a `names` mapping.
    try:
        doc = yaml.safe_load(text)
        names = doc.get("names") if isinstance(doc, dict) else None
        if isinstance(names, dict):
            # names may be {int: str} or {str: str}
            return {int(k): str(v) for k, v in names.items()}
    except yaml.YAMLError:
        pass

    out: dict[int, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, rest = line.partition(" ")
        try:
            cid = int(head)
        except ValueError:
            continue
        name = rest.strip()
        if name:
            out[cid] = name
    return out


# ---- label scan -----------------------------------------------------------


def scan_labels_for_classes(
    labels_dir: Path,
    id_to_name: dict[int, str],
    classes_to_check: set[str],
) -> dict[str, set[str]]:
    """Return ``{image_id: {class_name, ...}}`` limited to ``classes_to_check``.

    Image id is the label filename stem. Rows whose class_id is absent from
    ``id_to_name`` are skipped with a debug log.
    """
    needed: dict[str, set[str]] = {}
    unknown_ids: set[int] = set()
    n_files = 0

    for entry in os.scandir(labels_dir):
        if not entry.name.endswith(".txt"):
            continue
        n_files += 1
        stem = entry.name[:-4]
        classes: set[str] = set()
        try:
            with open(entry.path) as f:
                for raw in f:
                    head = raw.split(" ", 1)[0]
                    if not head:
                        continue
                    try:
                        cid = int(head)
                    except ValueError:
                        continue
                    name = id_to_name.get(cid)
                    if name is None:
                        unknown_ids.add(cid)
                        continue
                    if name in classes_to_check:
                        classes.add(name)
        except OSError as e:
            log.warning("label read failed %s: %s", entry.path, e)
            continue
        if classes:
            needed[stem] = classes

    if unknown_ids:
        log.warning("label scan: %d unknown class ids (e.g. %s) — skipped",
                    len(unknown_ids), sorted(unknown_ids)[:5])
    log.info("label scan: %d label files -> %d images needing validation "
             "(classes_to_check=%s)",
             n_files, len(needed), sorted(classes_to_check))
    return needed


# ---- per-class precomputed prompt bundle ----------------------------------


@dataclass
class ClassBundle:
    name: str
    positive: list[str]
    negative: list[str]
    junk: list[str]
    # One (n_pos + n_neg + n_junk, dim) fp16 matrix so scoring is a single
    # matmul per batch per class. Slicing below recovers pos/neg/junk groups.
    all_emb: np.ndarray
    num_pos: int
    num_neg: int
    num_junk: int
    thr: object  # Thresholds


def build_bundles(
    model: SigLIPModel,
    cfg: ValidationConfig,
) -> dict[str, ClassBundle]:
    bundles: dict[str, ClassBundle] = {}
    for name in cfg.classes_to_check:
        cp: ClassPrompts = cfg.prompts[name]
        all_prompts = list(cp.positive) + list(cp.negative) + list(cp.junk)
        all_emb = model.get_text_embeddings(all_prompts)
        bundles[name] = ClassBundle(
            name=name,
            positive=cp.positive,
            negative=cp.negative,
            junk=cp.junk,
            all_emb=all_emb,
            num_pos=len(cp.positive),
            num_neg=len(cp.negative),
            num_junk=len(cp.junk),
            thr=cp.thr,
        )
        log.info("class %s: %d positive / %d negative / %d junk prompts",
                 name, len(cp.positive), len(cp.negative), len(cp.junk))
    return bundles


# ---- scoring + threshold rule ---------------------------------------------


def apply_threshold_rule(
    pos_max: np.ndarray,
    neg_max: np.ndarray | None,
    junk_max: np.ndarray | None,
    thr,
) -> tuple[np.ndarray, list[str | None]]:
    """Return (pass_mask, fail_reason_per_row).

    Mirrors data_miner.modules.frame_filter._apply_filter exactly.
    ``fail_reason`` is the first rule a failing row tripped, so accepts get
    ``None``.
    """
    n = pos_max.shape[0]
    pos_ok = pos_max > thr.positive_thr
    neg_ok = np.ones(n, dtype=bool)
    junk_ok = np.ones(n, dtype=bool)
    neg_margin_ok = np.ones(n, dtype=bool)
    junk_margin_ok = np.ones(n, dtype=bool)

    if neg_max is not None:
        neg_ok = neg_max < thr.negative_thr
        neg_margin_ok = (pos_max - neg_max) > thr.pos_neg_margin_thr
    if junk_max is not None:
        junk_ok = junk_max < thr.junk_thr
        junk_margin_ok = (pos_max - junk_max) > thr.pos_junk_margin_thr

    passed = pos_ok & neg_ok & junk_ok & neg_margin_ok & junk_margin_ok
    reasons: list[str | None] = []
    for i in range(n):
        if passed[i]:
            reasons.append(None)
        elif not pos_ok[i]:
            reasons.append("pos<thr")
        elif not neg_ok[i]:
            reasons.append("neg>=thr")
        elif not neg_margin_ok[i]:
            reasons.append("neg_margin")
        elif not junk_ok[i]:
            reasons.append("junk>=thr")
        else:
            reasons.append("junk_margin")
    return passed, reasons


def score_batch_for_class(
    model: SigLIPModel,
    img_emb: np.ndarray,   # (B, dim) fp16
    bundle: ClassBundle,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, list[str]]:
    """Single matmul (B, dim) @ (k, dim).T for all pos/neg/junk prompts of
    this class, then slice and take max per group.

    Returns (pos_max, neg_max_or_None, junk_max_or_None, best_positive_names).
    """
    scores = model.compute_similarity(img_emb, bundle.all_emb)   # (B, k)
    np_ = bundle.num_pos
    nn = bundle.num_neg
    nj = bundle.num_junk

    pos_scores = scores[:, :np_]
    best_idx = pos_scores.argmax(axis=1)
    pos_max = pos_scores[np.arange(pos_scores.shape[0]), best_idx]

    neg_max = scores[:, np_:np_ + nn].max(axis=1) if nn else None
    junk_max = scores[:, np_ + nn:np_ + nn + nj].max(axis=1) if nj else None

    best_names = [bundle.positive[int(i)] for i in best_idx]
    return pos_max, neg_max, junk_max, best_names


# ---- top-K report heaps ---------------------------------------------------


class TopKHeap:
    """Fixed-size max heap by score: keep the *highest* ``k`` entries seen."""

    def __init__(self, k: int):
        self.k = k
        self._h: list[tuple[float, str, dict]] = []
        self._tie = 0

    def offer(self, score: float, image_id: str, payload: dict) -> None:
        self._tie += 1
        entry = (score, self._tie, image_id, payload)
        if len(self._h) < self.k:
            heapq.heappush(self._h, entry)
        else:
            heapq.heappushpop(self._h, entry)

    def dump(self) -> list[dict]:
        out = []
        for score, _tie, image_id, payload in sorted(self._h, reverse=True):
            out.append({"image_id": image_id, "pos_score_max": float(score),
                        **payload})
        return out


# ---- main runner ----------------------------------------------------------


def existing_pairs(tbl_uri: Path, table_name: str) -> set[tuple[str, str]]:
    """Return the set of (image_id, class_name) pairs already in the verdict
    table, or an empty set if the table doesn't exist yet.
    """
    import lancedb
    db = lancedb.connect(str(tbl_uri))
    if table_name not in db.table_names():
        return set()
    tbl = db.open_table(table_name)
    if tbl.count_rows() == 0:
        return set()
    out: set[tuple[str, str]] = set()
    ds = tbl.to_lance()
    reader = ds.scanner(columns=["image_id", "class_name"]).to_reader()
    for rb in reader:
        ids = rb.column("image_id").to_pylist()
        names = rb.column("class_name").to_pylist()
        out.update(zip(ids, names))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, required=True,
                    help="Path to label validation YAML")
    ap.add_argument("--lance-uri", type=str,
                    default="/mnt/data/data_miner_lance/laion_good3")
    ap.add_argument("--embed-table", type=str, default="embeddings")
    ap.add_argument("--labels-dir", type=str,
                    default="output/auto_annotation_v4/laion_good3/labels_standalone")
    ap.add_argument("--classes-yaml", type=str,
                    default="output/auto_annotation_v4/laion_good3/data.yaml",
                    help="data.yaml or classes.txt; only class-id -> name map is used")
    ap.add_argument("--out-table", type=str, default="label_validation")
    ap.add_argument("--report-jsonl", type=str,
                    default="output/label_validation/good3_report.jsonl")
    ap.add_argument("--report-topk", type=int, default=50)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-rows", type=int, default=50_000,
                    help="Rows per Lance scanner batch")
    ap.add_argument("--buffer-rows", type=int, default=4096,
                    help="Verdict-writer flush size")
    ap.add_argument("--resume", action="store_true",
                    help="Skip (image_id, class) pairs already in out-table")
    ap.add_argument("--dry-run", action="store_true",
                    help="Scan labels + build needed map, print stats, exit")
    ap.add_argument("--score-only", action="store_true",
                    help="Skip threshold filtering; write every (image, class) "
                         "pair with verdict='score' and a populated fail_reason "
                         "of None. Useful for picking thresholds empirically.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N matching images (0 = all)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    id_to_name = load_id_to_name(Path(args.classes_yaml))

    # Fail fast if any YAML class name isn't in the dataset's class map.
    unknown_cfg_classes = set(cfg.classes_to_check) - set(id_to_name.values())
    if unknown_cfg_classes:
        raise ValueError(
            f"classes_to_check includes names not in classes map: "
            f"{sorted(unknown_cfg_classes)}"
        )

    # 1. Scan labels -> needed_by_image
    needed = scan_labels_for_classes(
        Path(args.labels_dir), id_to_name, set(cfg.classes_to_check),
    )
    per_class_count: dict[str, int] = defaultdict(int)
    for s in needed.values():
        for n in s:
            per_class_count[n] += 1
    for name in cfg.classes_to_check:
        log.info("  %s: %d images need validation", name,
                 per_class_count.get(name, 0))

    if args.dry_run:
        log.info("dry-run: stopping before any GPU work")
        return 0

    # 2. Resume set
    lance_uri = Path(args.lance_uri)
    skip = existing_pairs(lance_uri, args.out_table) if args.resume else set()
    if args.resume:
        log.info("resume: skipping %d (image_id, class) pairs already in %s",
                 len(skip), args.out_table)

    # 3. Model + prompt embeddings
    model_id = SIGLIP2_MODELS.get(cfg.model, cfg.model)  # accept alias or full id
    log.info("loading SigLIP model: %s  (device=%s)", model_id, args.device)
    model = SigLIPModel(model_id=model_id, device_map=args.device)
    model.load()
    bundles = build_bundles(model, cfg)

    # 4. Writer for the verdict table
    writer = LanceEmbeddingWriter(
        uri=str(lance_uri),
        table=args.out_table,
        schema=build_verdict_schema(),
        buffer_rows=args.buffer_rows,
        create_if_missing=True,
    )

    # 5. Top-K heaps per (class, verdict)
    top_pass: dict[str, TopKHeap] = {n: TopKHeap(args.report_topk) for n in cfg.classes_to_check}
    top_fail: dict[str, TopKHeap] = {n: TopKHeap(args.report_topk) for n in cfg.classes_to_check}

    # 6. Stream the embeddings table
    import lancedb
    db = lancedb.connect(str(lance_uri))
    if args.embed_table not in db.table_names():
        raise FileNotFoundError(f"embeddings table missing: {lance_uri}/{args.embed_table}")
    emb_tbl = db.open_table(args.embed_table)
    log.info("scanning %s/%s (%d rows) in batches of %d",
             lance_uri, args.embed_table, emb_tbl.count_rows(), args.batch_rows)

    ds = emb_tbl.to_lance()
    scanner = ds.scanner(
        columns=["image_id", "siglip2_embedding"],
        batch_size=args.batch_rows,
    )
    reader = scanner.to_reader()

    t0 = time.perf_counter()
    n_rows_scanned = 0
    n_verdicts_emitted = 0
    n_images_matched = 0
    n_batches = 0
    log_every_batches = 5
    stopped_early = False
    score_dist: dict[str, list[float]] = {}
    pass_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}

    for rb in reader:
        n_batches += 1
        ids = rb.column("image_id").to_pylist()
        # Lance fixed_size_list column to (B, dim) fp16 numpy
        emb_col = rb.column("siglip2_embedding")
        emb_np = emb_col.values.to_numpy(zero_copy_only=False).reshape(
            len(ids), -1,
        ).astype(np.float16, copy=False)

        n_rows_scanned += len(ids)

        # Per-class scoring: collect row indices that need *this* class.
        # Skip pairs already in the resume set.
        per_class_indices: dict[str, list[int]] = defaultdict(list)
        for local_i, image_id in enumerate(ids):
            classes = needed.get(image_id)
            if not classes:
                continue
            for cname in classes:
                if args.resume and (image_id, cname) in skip:
                    continue
                per_class_indices[cname].append(local_i)

        if not per_class_indices:
            continue

        # Count unique images matched in this batch.
        n_images_matched += len({i for idxs in per_class_indices.values() for i in idxs})

        now = datetime.now(timezone.utc)
        batch_verdict_rows: list[dict] = []

        for cname, row_indices in per_class_indices.items():
            bundle = bundles[cname]
            sub = emb_np[row_indices]
            pos_max, neg_max, junk_max, best_names = score_batch_for_class(
                model, sub, bundle,
            )
            if args.score_only:
                passed = np.ones(pos_max.shape[0], dtype=bool)
                reasons = [None] * pos_max.shape[0]
            else:
                passed, reasons = apply_threshold_rule(
                    pos_max, neg_max, junk_max, bundle.thr,
                )
            # Collect raw scores for percentile log at end of run
            score_dist.setdefault(cname, []).extend(pos_max.tolist())
            pass_counts[cname] = pass_counts.get(cname, 0) + int(passed.sum())
            total_counts[cname] = total_counts.get(cname, 0) + int(passed.shape[0])

            # Emit rows + update top-K heaps.
            neg_scalar = neg_max if neg_max is not None else np.zeros_like(pos_max)
            junk_scalar = junk_max if junk_max is not None else np.zeros_like(pos_max)

            for k, local_i in enumerate(row_indices):
                image_id = ids[local_i]
                if args.score_only:
                    verdict = "score"
                else:
                    verdict = "pass" if passed[k] else "fail"
                row = {
                    "image_id": image_id,
                    "class_name": cname,
                    "verdict": verdict,
                    "fail_reason": reasons[k],
                    "pos_score_max": float(pos_max[k]),
                    "neg_score_max": float(neg_scalar[k]),
                    "junk_score_max": float(junk_scalar[k]),
                    "best_positive": best_names[k],
                    "created_at": now,
                }
                batch_verdict_rows.append(row)
                # In --score-only mode there are no "fails"; put everything in
                # top_pass so the report JSONL still highlights top scorers.
                if args.score_only or passed[k]:
                    heap = top_pass[cname]
                else:
                    heap = top_fail[cname]
                heap.offer(
                    float(pos_max[k]), image_id,
                    {
                        "class_name": cname,
                        "verdict": row["verdict"],
                        "fail_reason": row["fail_reason"],
                        "neg_score_max": row["neg_score_max"],
                        "junk_score_max": row["junk_score_max"],
                        "best_positive": row["best_positive"],
                    },
                )

        if batch_verdict_rows:
            writer.extend(batch_verdict_rows)
            n_verdicts_emitted += len(batch_verdict_rows)

        if n_batches % log_every_batches == 0:
            dt = time.perf_counter() - t0
            log.info("scanned %d rows | matched %d imgs | verdicts %d | %.0f rows/s",
                     n_rows_scanned, n_images_matched, n_verdicts_emitted,
                     n_rows_scanned / max(dt, 1e-6))

        if args.limit and n_images_matched >= args.limit:
            log.info("--limit reached (%d matched images); stopping scan", args.limit)
            stopped_early = True
            break

    writer.close()
    dt = time.perf_counter() - t0
    log.info("done: %d rows scanned | %d images matched | %d verdicts "
             "written in %.1fs%s",
             n_rows_scanned, n_images_matched, n_verdicts_emitted, dt,
             " (stopped early)" if stopped_early else "")

    # Per-class score percentile summary — helpful for tuning thresholds.
    for cname in cfg.classes_to_check:
        scores = score_dist.get(cname, [])
        if not scores:
            log.info("class %s: no matches scored", cname)
            continue
        arr = np.asarray(scores, dtype=np.float64)
        pct = np.percentile(arr, [5, 25, 50, 75, 90, 95, 99])
        total = total_counts.get(cname, 0)
        passed = pass_counts.get(cname, 0)
        log.info(
            "class %s: n=%d pass=%d (%.1f%%) | "
            "pos_score pcts p5=%.3e p25=%.3e p50=%.3e p75=%.3e p90=%.3e p95=%.3e p99=%.3e",
            cname, total, passed, 100.0 * passed / max(total, 1),
            *pct,
        )

    # 7. Dump top-K report
    report_path = Path(args.report_jsonl)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        for cname in cfg.classes_to_check:
            for entry in top_pass[cname].dump():
                f.write(json.dumps({"bucket": "top_pass", **entry}) + "\n")
            for entry in top_fail[cname].dump():
                f.write(json.dumps({"bucket": "top_fail", **entry}) + "\n")
    log.info("report written to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
