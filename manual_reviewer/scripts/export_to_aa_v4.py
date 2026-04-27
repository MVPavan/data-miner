"""Pull Label Studio completions and write them back to aa_v4 ``pipeline.db``.

Pull-mode CLI (no webhook listener required for v1). Two input modes:

- ``--in-file <path.json>``: read completions from a JSON file (LS export
  → JSON-MIN or JSON format). Useful for bake testing and air-gapped
  workflows.
- ``--ls-url --ls-token --ls-project``: query LS export endpoint with an
  optional ``--since=<unix-ts>`` cutoff to fetch only new completions.

For each completion the script:

1. Resolves the aa_v4 ``image_id`` from ``task.data.image_id``.
2. Re-reads the original task to recover the seeded predictions and
   ``ghost_drops`` so the parser can classify edits (relabeled vs added vs
   kept_dropped). Falls back to no-seeding if the original prediction is
   absent, which downgrades every region to ``added``.
3. Builds a ``HumanReviewResult`` and writes it via ``write_human_review``.
4. Optionally appends a ``human_review`` block to ``traces/{image_id}.json``
   so the file-based audit log stays consistent with the DB.
5. Optionally rewrites ``labels/{image_id}.txt`` from corrected boxes so
   downstream YOLO consumers see the human truth (use ``--rewrite-yolo``).

Usage::

    python -m manual_reviewer.scripts.export_to_aa_v4 \\
        --db /jobs/run_42/pipeline.db \\
        --traces-dir /jobs/run_42/traces \\
        --in-file ls_export.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from data_miner.auto_annotation_v4.configs.contracts import HumanReviewResult

from manual_reviewer.pipeline_io import parse_ls_completion, write_human_review

logger = logging.getLogger("manual_reviewer.export_to_aa_v4")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    db_path = args.db.resolve()
    if not db_path.exists():
        logger.error("pipeline.db not found: %s", db_path)
        return 2

    completions = _load_completions(args)
    if not completions:
        logger.warning("no completions to process")
        return 1

    written = 0
    for ls_completion, task_data, predictions in completions:
        image_id = (task_data or {}).get("image_id")
        if not image_id:
            logger.warning("skipping completion %s: no image_id in task data", ls_completion.get("id"))
            continue
        seeded = _extract_seeded(predictions)
        ghost_ids = _extract_ghost_ids(task_data)
        try:
            result = parse_ls_completion(
                ls_completion,
                image_id=image_id,
                seeded_predictions=seeded,
                ghost_drop_ids=ghost_ids,
            )
        except Exception:  # noqa: BLE001
            logger.exception("parse_ls_completion failed for %s", image_id)
            continue
        try:
            write_human_review(db_path, result, config_hash=args.config_hash or "")
        except Exception:  # noqa: BLE001
            logger.exception("write_human_review failed for %s", image_id)
            continue
        written += 1

        if args.traces_dir:
            _append_trace(args.traces_dir, image_id, result)
        if args.rewrite_yolo and args.labels_dir:
            _rewrite_yolo_label(args.labels_dir, image_id, result, args.classes_file)

    logger.info("wrote %d human_review rows", written)
    return 0 if written else 1


def _load_completions(args: argparse.Namespace) -> list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]]:
    """Return [(ls_completion, task_data, seeded_predictions), ...] tuples."""
    if args.in_file:
        raw = json.loads(args.in_file.read_text(encoding="utf-8"))
        return list(_walk_export_file(raw))
    if args.ls_url:
        return _fetch_from_ls(args)
    raise SystemExit("provide either --in-file or --ls-url")


def _walk_export_file(raw: Any) -> list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]]:
    """Iterate LS-format JSON export.

    Standard export: ``[{ "id": ..., "data": {...},
                          "annotations": [{"result": [...]}],
                          "predictions": [{"result": [...]}]}]``
    JSON-MIN export: ``[{"id": ..., "image_id": ..., "bbox": [...]}, ...]`` —
    we don't support JSON-MIN here because it loses prediction history.
    """
    out: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    if not isinstance(raw, list):
        raise ValueError("expected a JSON array of LS task exports")
    for task in raw:
        if not isinstance(task, dict):
            continue
        data = task.get("data") or {}
        annotations = task.get("annotations") or []
        predictions = task.get("predictions") or []
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            if ann.get("was_cancelled") or ann.get("ground_truth") is False and ann.get("result") is None:
                continue
            out.append((ann, data, predictions))
    return out


def _fetch_from_ls(args: argparse.Namespace) -> list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]]:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is required for --ls-url mode") from exc
    if not args.ls_token or args.ls_project is None:
        raise SystemExit("--ls-url requires --ls-token and --ls-project")

    base = args.ls_url.rstrip("/")
    url = f"{base}/api/projects/{args.ls_project}/export"
    params: dict[str, Any] = {"exportType": "JSON"}
    headers = {"Authorization": f"Token {args.ls_token}"}
    with httpx.Client(timeout=args.ls_timeout, headers=headers) as client:
        resp = client.get(url, params=params)
        if resp.status_code >= 300:
            raise RuntimeError(f"LS export failed {resp.status_code}: {resp.text[:500]}")
        raw = resp.json()
    if args.since is not None:
        cutoff = float(args.since)
        raw = [t for t in raw if _task_after(t, cutoff)]
    return list(_walk_export_file(raw))


def _task_after(task: dict[str, Any], cutoff: float) -> bool:
    for ann in task.get("annotations") or []:
        ts = ann.get("updated_at") or ann.get("created_at")
        if isinstance(ts, str):
            try:
                from datetime import datetime
                t = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except ValueError:
                continue
            if t >= cutoff:
                return True
    return False


def _extract_seeded(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pull region results out of the LS predictions[] envelope."""
    if not predictions:
        return []
    first = predictions[0]
    if isinstance(first, dict):
        return list(first.get("result") or [])
    return []


def _extract_ghost_ids(task_data: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for entry in task_data.get("ghost_drops") or []:
        if isinstance(entry, dict):
            cid = entry.get("candidate_id")
            if isinstance(cid, str) and cid:
                out.add(cid)
    return out


def _append_trace(traces_dir: Path, image_id: str, result: HumanReviewResult) -> None:
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_path = traces_dir / f"{image_id}.json"
    block = {
        "stage": "human_review",
        "ts": time.time(),
        "data": json.loads(result.model_dump_json()),
    }
    if trace_path.exists():
        try:
            existing = json.loads(trace_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = None
        if isinstance(existing, list):
            existing.append(block)
            trace_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            return
        if isinstance(existing, dict):
            existing.setdefault("history", []).append(block)
            trace_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            return
    trace_path.write_text(json.dumps([block], indent=2), encoding="utf-8")


def _rewrite_yolo_label(
    labels_dir: Path,
    image_id: str,
    result: HumanReviewResult,
    classes_file: Path | None,
) -> None:
    """Rewrite ``labels/{image_id}.txt`` from corrections.

    Class id resolution: read names from ``classes_file`` (one name per line)
    if provided; otherwise leave ``class_id=0`` and emit a warning. Boxes
    written as ``class_id cx cy w h`` with normalized coordinates.
    """
    class_to_id: dict[str, int] = {}
    if classes_file and classes_file.exists():
        for i, ln in enumerate(classes_file.read_text(encoding="utf-8").splitlines()):
            name = ln.strip()
            if name:
                class_to_id[name] = i

    lines: list[str] = []
    for c in result.corrections:
        cls_id = class_to_id.get(c.class_name, 0)
        if c.class_name not in class_to_id and class_to_id:
            logger.warning("class %r not in %s — using 0", c.class_name, classes_file)
        cx = (c.bbox.x1 + c.bbox.x2) / 2.0
        cy = (c.bbox.y1 + c.bbox.y2) / 2.0
        w = c.bbox.x2 - c.bbox.x1
        h = c.bbox.y2 - c.bbox.y1
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write LS completions to aa_v4 pipeline.db as human_review")
    p.add_argument("--db", type=Path, required=True)
    p.add_argument("--traces-dir", type=Path, default=None, help="Append human_review block to traces/{id}.json")
    p.add_argument("--labels-dir", type=Path, default=None, help="Path to YOLO labels dir for --rewrite-yolo")
    p.add_argument("--classes-file", type=Path, default=None, help="classes.txt for --rewrite-yolo class id lookup")
    p.add_argument("--rewrite-yolo", action="store_true", help="Rewrite YOLO labels/{id}.txt from corrections")
    p.add_argument("--config-hash", default="", help="config_hash to record in stages row")

    src = p.add_mutually_exclusive_group()
    src.add_argument("--in-file", type=Path, default=None)
    src.add_argument("--ls-url", default=None)

    p.add_argument("--ls-token", default=None)
    p.add_argument("--ls-project", type=int, default=None)
    p.add_argument("--ls-timeout", type=float, default=60.0)
    p.add_argument("--since", type=float, default=None,
                   help="Unix timestamp; only fetch completions updated at/after this")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
