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
import os
import sys
from pathlib import Path
from typing import Any

from data_miner.annotation_io import append_human_review_trace, rewrite_yolo_label
from data_miner.auto_annotation_v4.configs.contracts import HumanReviewResult
from manual_reviewer import FINALIZE_MODEL_VERSION
from manual_reviewer.pipeline_io import parse_ls_completion, write_human_review

logger = logging.getLogger("manual_reviewer.export_to_aa_v4")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    db_path = args.db.resolve()
    if not db_path.exists():
        logger.error("pipeline.db not found: %s", db_path)
        return 2

    # Hard-fail at startup if YOLO rewrite is requested without a class
    # registry — silently writing every box as class 0 would corrupt the
    # downstream dataset and only show up at training time.
    if args.rewrite_yolo and args.labels_dir and not args.classes_file:
        logger.error(
            "--rewrite-yolo requires --classes-file; without a class registry "
            "every label would silently collapse to class 0"
        )
        return 2

    completions = _load_completions(args)
    if not completions:
        logger.warning("no completions to process")
        return 1

    written = 0
    errors = 0
    for ls_completion, task_data, predictions in completions:
        image_id = (task_data or {}).get("image_id")
        if not image_id:
            logger.warning(
                "skipping completion %s: no image_id in task data",
                ls_completion.get("id"),
            )
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
            errors += 1
            continue
        try:
            write_human_review(db_path, result, config_hash=args.config_hash or "")
        except Exception:  # noqa: BLE001
            logger.exception("write_human_review failed for %s", image_id)
            errors += 1
            continue
        written += 1

        if args.traces_dir:
            try:
                append_human_review_trace(args.traces_dir, image_id, result)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "append_trace failed for %s (DB write already done)", image_id
                )
                errors += 1
        if args.rewrite_yolo and args.labels_dir:
            if result.frame_state == "ambiguous_skip":
                logger.info(
                    "skip YOLO rewrite for %s: frame_state=ambiguous_skip",
                    image_id,
                )
            else:
                try:
                    rewrite_yolo_label(
                        args.labels_dir,
                        image_id,
                        result,
                        args.classes_file,
                        dry_run=args.dry_run,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "rewrite_yolo failed for %s (DB+trace already done)",
                        image_id,
                    )
                    errors += 1

    logger.info("wrote %d human_review rows (errors=%d)", written, errors)
    if errors:
        return 1
    return 0 if written else 1


def _load_completions(
    args: argparse.Namespace,
) -> list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]]:
    """Return [(ls_completion, task_data, seeded_predictions), ...] tuples."""
    if args.in_file:
        raw = json.loads(args.in_file.read_text(encoding="utf-8"))
        return list(_walk_export_file(raw))
    if args.ls_url:
        return _fetch_from_ls(args)
    raise SystemExit("provide either --in-file or --ls-url")


def _redact_token(s: str, token: str | None) -> str:
    if not token or not s:
        return s
    return s.replace(token, "<redacted>")


def _walk_export_file(
    raw: Any,
) -> list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]]:
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
            if ann.get("was_cancelled"):
                continue
            completion = dict(ann)
            completion.setdefault("task", task.get("id"))
            out.append((completion, data, predictions))
    return out


def _fetch_from_ls(
    args: argparse.Namespace,
) -> list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]]:
    """Page through ``/api/projects/{id}/tasks`` for tasks with annotations.

    The ``/api/projects/{id}/export`` endpoint returns predictions truncated to
    integer IDs, which strips the seeded prediction objects we need to classify
    edits as ``relabeled`` / ``edited`` rather than ``added``. The tasks
    endpoint returns predictions inline.
    """
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is required for --ls-url mode") from exc
    if not args.ls_token or args.ls_project is None:
        raise SystemExit("--ls-url requires --ls-token and --ls-project")

    base = args.ls_url.rstrip("/")
    headers = {"Authorization": f"Token {args.ls_token}"}
    url = f"{base}/api/projects/{args.ls_project}/tasks"

    raw: list[dict[str, Any]] = []
    page = 1
    page_size = 100
    with httpx.Client(timeout=args.ls_timeout, headers=headers) as client:
        while True:
            resp = client.get(url, params={"page": page, "page_size": page_size})
            if resp.status_code >= 300:
                body = _redact_token(resp.text[:500], args.ls_token)
                raise RuntimeError(f"LS tasks fetch failed {resp.status_code}: {body}")
            payload = resp.json()
            tasks = payload if isinstance(payload, list) else payload.get("tasks") or []
            if not tasks:
                break
            raw.extend(
                t for t in tasks if isinstance(t, dict) and (t.get("annotations") or [])
            )
            if len(tasks) < page_size:
                break
            page += 1

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
    """Pull the finalize-seeded region results out of LS ``predictions[]``.

    ``predictions[0]`` is unsafe with Phase B+ live: the ML backend stores
    its own predictions (smart_search re-runs, etc.) which can land in front
    of the finalize seed. The diff classifier in :func:`parse_ls_completion`
    needs the finalize baseline specifically to tag accepts as
    ``finalize`` / ``edited`` / ``relabeled`` (vs ``added``).
    """
    if not predictions:
        return []
    for pred in predictions:
        if not isinstance(pred, dict):
            continue
        if pred.get("model_version") == FINALIZE_MODEL_VERSION:
            return list(pred.get("result") or [])
    # Fallback: legacy exports without ``model_version`` set. Use the first
    # entry but log so operators can spot the drift.
    first = predictions[0]
    if isinstance(first, dict):
        if first.get("model_version"):
            logger.warning(
                "no predictions[] entry tagged %s; using model_version=%s "
                "as the finalize baseline",
                FINALIZE_MODEL_VERSION,
                first.get("model_version"),
            )
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
    """Compatibility wrapper for the shared human-review trace writer."""
    append_human_review_trace(traces_dir, image_id, result)


def _rewrite_yolo_label(
    labels_dir: Path,
    image_id: str,
    result: HumanReviewResult,
    classes_file: Path | None,
    *,
    dry_run: bool = False,
) -> None:
    """Compatibility wrapper for the shared YOLO label writer."""
    rewrite_yolo_label(labels_dir, image_id, result, classes_file, dry_run=dry_run)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Write LS completions to aa_v4 pipeline.db as human_review"
    )
    p.add_argument("--db", type=Path, required=True)
    p.add_argument(
        "--traces-dir",
        type=Path,
        default=None,
        help="Append human_review block to traces/{id}.json",
    )
    p.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Path to YOLO labels dir for --rewrite-yolo",
    )
    p.add_argument(
        "--classes-file",
        type=Path,
        default=None,
        help="classes.txt for --rewrite-yolo class id lookup",
    )
    p.add_argument(
        "--rewrite-yolo",
        action="store_true",
        help="Rewrite YOLO labels/{id}.txt from corrections",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Log YOLO rewrites without touching disk; DB and trace writes still occur",
    )
    p.add_argument(
        "--config-hash", default="", help="config_hash to record in stages row"
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--in-file", type=Path, default=None)
    src.add_argument("--ls-url", default=None)

    p.add_argument(
        "--ls-token",
        default=os.environ.get("LS_TOKEN"),
        help="LS API token; defaults to $LS_TOKEN env var (preferred — avoids exposure in `ps`)",
    )
    p.add_argument("--ls-project", type=int, default=None)
    p.add_argument("--ls-timeout", type=float, default=60.0)
    p.add_argument(
        "--since",
        type=float,
        default=None,
        help="Unix timestamp; only fetch completions updated at/after this. "
        "Mutually exclusive with --in-file (file mode reads everything)",
    )
    args = p.parse_args(argv)
    if args.in_file and args.since is not None:
        p.error(
            "--since is incompatible with --in-file (file mode reads the export wholesale)"
        )
    return args


if __name__ == "__main__":
    sys.exit(main())
