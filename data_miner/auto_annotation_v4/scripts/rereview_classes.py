"""Targeted VLM re-review for a subset of final classes.

Reads an existing job's pipeline.db read-only, re-runs the evaluate-stage
VLM call on every final annotation whose class is in ``--classes``, and
writes the results to ``<job_dir>/<out_subdir>/`` WITHOUT mutating
production state (labels/, pipeline.db, review/ are untouched).

Revert = ``rm -rf <job_dir>/<out_subdir>/``.

Example:
    python -m data_miner.auto_annotation_v4.scripts.rereview_classes \\
        output/auto_annotation_v4/loco_unannotated_full_sam_filtered_detect \\
        --classes forklift,palletjack \\
        --override data_miner/auto_annotation_v4/configs/overrides/loco_rereview_fp.yaml \\
        --vlm-url http://localhost:8956/v1 \\
        --vlm-model Qwen/Qwen3.5-27B-FP8 \\
        --concurrency 32

Re-uses EvaluateWorker._classify_one / _resolve_verdicts as library calls
(zero duplication) — the worker is instantiated with a dummy DB since we
never save checkpoints or claim work.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
from PIL import Image

from ..configs import (
    AutoAnnotationV4Config,
    BoundingBox,
    Candidate,
    CandidateStatus,
)
from ..configs.loader import load_config
from ..stages.evaluate import EvaluateWorker

logger = logging.getLogger("rereview")


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job_dir", type=Path, help="Output dir containing pipeline.db + labels/")
    p.add_argument(
        "--classes", required=True,
        help="Comma-separated list of class names to re-review (e.g. forklift,palletjack)",
    )
    p.add_argument(
        "--override", type=Path, required=True,
        help="YAML override path (merged on top of default + class_config).",
    )
    p.add_argument("--vlm-url", default="http://localhost:8956/v1")
    p.add_argument("--vlm-model", default="Qwen/Qwen3.5-27B-FP8")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument(
        "--out-subdir", default="rereview",
        help="Subdirectory under <job_dir> to write results into.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Only re-review this many images (dry-run). Candidates inside them all still run.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _select_image_candidates(
    db_path: Path, target_classes: set[str], limit: int | None,
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Return [(image_id, image_path, [candidates_to_rereview])].

    Each candidate dict carries: candidate_id, class_name, class_id, bbox,
    confidence, source_model. Non-target rows are handled later (they're
    read directly from labels/<image>.txt to preserve them unchanged).
    """
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    # image_id -> image_path
    paths = {
        row[0]: row[1]
        for row in con.execute("SELECT image_id, image_path FROM image_meta")
    }
    out: list[tuple[str, str, list[dict[str, Any]]]] = []
    for image_id, data in con.execute(
        "SELECT image_id, data FROM stages WHERE stage='finalize'"
    ):
        d = json.loads(data)
        targets = [
            a for a in (d.get("final_annotations") or [])
            if a.get("class_name") in target_classes
        ]
        if not targets:
            continue
        ipath = paths.get(image_id)
        if not ipath:
            logger.warning("no image_path in image_meta for %s; skipping", image_id)
            continue
        out.append((image_id, ipath, targets))
        if limit is not None and len(out) >= limit:
            break
    con.close()
    return out


def _final_annotation_to_candidate(a: dict[str, Any]) -> Candidate:
    """Rebuild a Candidate from a stored FinalAnnotation dict.

    The VLM call only reads bbox + class_name + candidate_id, so a minimal
    reconstruction is enough — agreement / notes / metadata irrelevant.
    """
    bb = a["bbox"]
    return Candidate(
        candidate_id=a["candidate_id"],
        class_name=a["class_name"],
        label=a["class_name"],
        source_model=a.get("source_model") or "sam3_dart",
        expression=a["class_name"],
        bbox=BoundingBox(x1=bb["x1"], y1=bb["y1"], x2=bb["x2"], y2=bb["y2"]),
        score=float(a.get("confidence", 1.0)),
        agreement=1,
        agreeing_models=[a.get("source_model") or "sam3_dart"],
        status=CandidateStatus.PROPOSED,
        metadata={"rereview_src": "finalize"},
    )


def _read_orig_labels(labels_dir: Path, image_id: str) -> list[str]:
    """Return raw YOLO lines from the original labels/<image>.txt.

    Used to preserve non-target rows in the rereview output.
    """
    p = labels_dir / f"{image_id}.txt"
    if not p.exists():
        return []
    return [ln.rstrip() for ln in p.read_text().splitlines() if ln.strip()]


def _bbox_to_yolo_line(class_id: int, bb: BoundingBox, score: float | None = None) -> str:
    cx = (bb.x1 + bb.x2) / 2
    cy = (bb.y1 + bb.y2) / 2
    w = bb.x2 - bb.x1
    h = bb.y2 - bb.y1
    base = f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    return f"{base} {score:.6f}" if score is not None else base


def _orig_row_matches_candidate(
    row: str, target_class_ids: set[int],
) -> bool:
    """True if this YOLO row's class id is in target set."""
    parts = row.split()
    if not parts:
        return False
    try:
        return int(parts[0]) in target_class_ids
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Dummy DB — EvaluateWorker stores this ref but we never call DB methods.
# ---------------------------------------------------------------------------


class _NullDB:
    """No-op CheckpointDB stand-in. Script never saves / loads / claims."""
    # Attributes the worker base may touch:
    db_path = Path("/dev/null")
    max_retries = 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _rereview_one_image(
    worker: EvaluateWorker,
    sem: asyncio.Semaphore,
    image_path: str,
    cands: list[Candidate],
    class_to_group: dict[str, str],
) -> dict[str, Any]:
    """Run VLM + routing for one image's target candidates.

    Returns {verdicts, accepted_ids, review_ids, rejected_ids, relabels,
    drops, vlm_tokens}. Caller must have set ``worker._session`` beforehand.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        logger.warning("failed to open %s: %s", image_path, exc)
        return {
            "verdicts": [], "accepted": [], "review": [], "rejected": [],
            "relabels": {}, "drops": [], "tokens": 0, "calls": 0,
            "error": f"image_open:{exc}",
        }

    tasks = [
        worker._classify_one(sem, image, c, class_to_group) for c in cands
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    from ..configs import VLMVerdict  # local import avoids cycle at module load
    verdicts: list[VLMVerdict] = []
    tokens = 0
    calls = 0
    malformed_ids: list[str] = []
    for cand, r in zip(cands, results):
        if isinstance(r, Exception) or r is None:
            # HTTP failure / timeout: keep original annotation as-is (skip).
            continue
        if r.get("malformed"):
            tokens += r.get("tokens", 0); calls += 1
            malformed_ids.append(r["candidate_id"])
            continue
        calls += 1
        tokens += r.get("tokens", 0)
        verdicts.append(r["verdict"])

    accepted, review, rejected, relabels, drops = worker._resolve_verdicts(verdicts, cands)
    # Treat malformed like rejects for routing purposes.
    rejected = list(rejected) + malformed_ids
    return {
        "verdicts": verdicts, "accepted": accepted, "review": review,
        "rejected": rejected, "relabels": relabels, "drops": drops,
        "tokens": tokens, "calls": calls,
    }


async def _main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    job_dir = args.job_dir.resolve()
    db_path = job_dir / "pipeline.db"
    labels_dir = job_dir / "labels"
    out_dir = job_dir / args.out_subdir
    out_labels = out_dir / "labels"
    out_review = out_dir / "review"

    if not db_path.exists():
        logger.error("pipeline.db not found at %s", db_path); return 2
    if not labels_dir.exists():
        logger.error("labels dir not found at %s", labels_dir); return 2
    out_labels.mkdir(parents=True, exist_ok=True)
    out_review.mkdir(parents=True, exist_ok=True)

    target_classes = {c.strip() for c in args.classes.split(",") if c.strip()}
    if not target_classes:
        logger.error("--classes must be non-empty"); return 2
    logger.info("target classes: %s", sorted(target_classes))

    # Load merged config (override merges on top of defaults). Also patch
    # VLM server from CLI for reproducibility when the override doesn't
    # mention it.
    overrides_cli = [
        f"servers.vlm.url={args.vlm_url}",
        f"servers.vlm.model={args.vlm_model}",
        f"evaluate.concurrency={args.concurrency}",
    ]
    cfg: AutoAnnotationV4Config = load_config(
        user_config=str(args.override), overrides=overrides_cli,
    )
    # class_name -> global class id (what YOLO labels use on disk).
    cls_id: dict[str, int] = {name: c.id for name, c in cfg.classes.items()}
    target_class_ids = {cls_id[c] for c in target_classes if c in cls_id}
    if len(target_class_ids) != len(target_classes):
        missing = target_classes - set(cls_id)
        logger.error("unknown target classes: %s", sorted(missing)); return 2

    # Serialise the effective override so the run is reproducible later.
    (out_dir / "config_used.yaml").write_text(args.override.read_text())

    # Select work
    work = _select_image_candidates(db_path, target_classes, args.limit)
    total_cands = sum(len(c) for _, _, c in work)
    logger.info(
        "images with target candidates: %d  |  total target candidates: %d",
        len(work), total_cands,
    )
    if not work:
        logger.info("nothing to re-review"); return 0

    # Build worker. Pass a null DB; script never saves checkpoints.
    worker = EvaluateWorker(cfg, _NullDB(), output_writer=None, worker_id="rereview")  # type: ignore[arg-type]
    worker.logger = logger
    class_to_group = worker._build_class_to_group()

    # Summary accumulators
    ran_images = 0
    vlm_calls_total = 0
    vlm_tokens_total = 0
    outcome_counts: dict[str, int] = {
        "accepted": 0, "review": 0, "rejected": 0, "relabel": 0, "skip": 0,
    }
    relabel_pairs: dict[tuple[str, str], int] = {}
    t0 = time.monotonic()

    verdicts_path = out_dir / "verdicts.jsonl"
    verdicts_f = verdicts_path.open("w", buffering=1)

    # Two bounds:
    #   - `vlm_sem` caps IN-FLIGHT VLM calls across the whole run (this is
    #     what --concurrency controls).
    #   - `image_sem` caps how many images are being actively processed
    #     at once (image load + per-candidate classify + per-image write).
    #     Without this, fan-out of ALL images at once would OOM on PIL
    #     buffers. 2× --concurrency keeps enough per-candidate tasks queued
    #     against the VLM semaphore to saturate it (avg ~2 cand/image).
    vlm_sem = asyncio.Semaphore(max(1, args.concurrency))
    image_sem = asyncio.Semaphore(max(4, args.concurrency * 2))
    state_lock = asyncio.Lock()

    async with aiohttp.ClientSession() as session:
        worker._session = session  # set once; _classify_one reads from here

        async def _process_image(image_id: str, image_path: str, targets: list[dict]) -> None:
            nonlocal vlm_calls_total, vlm_tokens_total, ran_images
            async with image_sem:
                cands = [_final_annotation_to_candidate(a) for a in targets]
                res = await _rereview_one_image(
                    worker, vlm_sem, image_path, cands, class_to_group,
                )

            # ---- build rereview label rows (no shared state yet) ----
            acc_ids = set(res["accepted"])
            rev_ids = set(res["review"])
            rej_ids = set(res["rejected"])
            relabels = res["relabels"]
            orig_rows = _read_orig_labels(labels_dir, image_id)
            kept_rows: list[str] = [
                r for r in orig_rows
                if not _orig_row_matches_candidate(r, target_class_ids)
            ]
            review_rows: list[str] = []
            per_cand_deltas: dict[str, int] = {
                "accepted": 0, "review": 0, "rejected": 0, "relabel": 0, "skip": 0,
            }
            per_cand_relabel_pairs: list[tuple[str, str]] = []
            verdict_lines: list[str] = []

            for c in cands:
                cid = c.candidate_id
                if cid in acc_ids:
                    new_class = relabels.get(cid, c.class_name)
                    gcid = cls_id.get(new_class)
                    if gcid is None:
                        per_cand_deltas["rejected"] += 1
                    else:
                        kept_rows.append(_bbox_to_yolo_line(gcid, c.bbox, score=c.score))
                        per_cand_deltas["accepted"] += 1
                        if new_class != c.class_name:
                            per_cand_deltas["relabel"] += 1
                            per_cand_relabel_pairs.append((c.class_name, new_class))
                elif cid in rev_ids:
                    new_class = relabels.get(cid, c.class_name)
                    gcid = cls_id.get(new_class, cls_id.get(c.class_name))
                    if gcid is not None:
                        review_rows.append(_bbox_to_yolo_line(gcid, c.bbox, score=c.score))
                    per_cand_deltas["review"] += 1
                    if cid in relabels:
                        per_cand_deltas["relabel"] += 1
                        per_cand_relabel_pairs.append((c.class_name, relabels[cid]))
                elif cid in rej_ids:
                    per_cand_deltas["rejected"] += 1
                else:
                    gcid = cls_id.get(c.class_name)
                    if gcid is not None:
                        kept_rows.append(_bbox_to_yolo_line(gcid, c.bbox, score=c.score))
                    per_cand_deltas["skip"] += 1

                v = next((v for v in res["verdicts"] if v.candidate_id == cid), None)
                outcome = (
                    "accepted" if cid in acc_ids else
                    "review" if cid in rev_ids else
                    "rejected" if cid in rej_ids else
                    "skip"
                )
                verdict_lines.append(json.dumps({
                    "image_id": image_id,
                    "candidate_id": cid,
                    "orig_class": c.class_name,
                    "detected_class": v.detected_class if v else None,
                    "class_confidence": v.class_confidence if v else None,
                    "bbox_score": v.bbox_score if v else None,
                    "object_complete": v.object_complete if v else None,
                    "reasoning": v.reasoning if v else None,
                    "outcome": outcome,
                    "relabel_to": relabels.get(cid),
                }))

            # Per-image files are unique paths; safe to write without a lock.
            (out_labels / f"{image_id}.txt").write_text(
                "\n".join(kept_rows) + ("\n" if kept_rows else "")
            )
            if review_rows:
                (out_review / f"{image_id}.txt").write_text("\n".join(review_rows) + "\n")

            # ---- merge into shared state under a lock ----
            async with state_lock:
                vlm_calls_total += res["calls"]
                vlm_tokens_total += res["tokens"]
                for k, v in per_cand_deltas.items():
                    outcome_counts[k] += v
                for pair in per_cand_relabel_pairs:
                    relabel_pairs[pair] = relabel_pairs.get(pair, 0) + 1
                for line in verdict_lines:
                    verdicts_f.write(line + "\n")
                ran_images += 1
                if ran_images % 50 == 0:
                    elapsed = time.monotonic() - t0
                    logger.info(
                        "progress %d/%d images  calls=%d tokens=%d  rate=%.1f cand/s",
                        ran_images, len(work), vlm_calls_total, vlm_tokens_total,
                        (outcome_counts["accepted"] + outcome_counts["review"] + outcome_counts["rejected"])
                        / max(elapsed, 1e-6),
                    )

        await asyncio.gather(*(
            _process_image(iid, ipath, targets) for iid, ipath, targets in work
        ))

    verdicts_f.close()
    elapsed = time.monotonic() - t0

    summary = {
        "job_dir": str(job_dir),
        "target_classes": sorted(target_classes),
        "images_rereviewed": ran_images,
        "candidates_total": total_cands,
        "vlm_calls": vlm_calls_total,
        "vlm_tokens": vlm_tokens_total,
        "wall_seconds": round(elapsed, 1),
        "outcomes": outcome_counts,
        "relabel_pairs": {f"{a} -> {b}": n for (a, b), n in sorted(relabel_pairs.items(), key=lambda x: -x[1])},
        "override_yaml": str(args.override),
        "vlm_url": args.vlm_url,
        "vlm_model": args.vlm_model,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    logger.info("done. wrote %s", out_dir)
    logger.info("summary: %s", json.dumps(summary, indent=2))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
