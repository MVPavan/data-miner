"""Build Label Studio tasks directly from a YOLO dataset.

Sidesteps ``pipeline.db`` for the case where a YOLO export (e.g. the
``output/dataset_selection/...`` tree) is the canonical truth and
reviewers should refine those labels in LS without first re-running
aa_v4 stages. The label .txt files become LS predictions; reviewers
edit/extend; ``export_to_aa_v4.py`` is not the export path here —
write a YOLO exporter from LS completions when the review pass is done.

Layout consumed::

    <DATASET_ROOT>/
      classes.txt           # one class per line; canonical label palette
      yolo/labels/<stem>.txt  # YOLO rows: class_id cx cy w h [score]
                              # the file set defines the import set
      stem_to_path.json     # {<stem>: "/abs/path/to/image.jpg", ...}

Class-id mapping note: aa_v4's YOLO exporter writes ``class_id`` using
the **global class_registry id** (which matches COCO's 80-class layout
plus extras), not the position in ``classes.txt``. For example, ``head``
has registry id ``35`` but is at position ``20`` in this dataset's
``classes.txt``. Pass ``--aav4-config <path/to/config.yaml>`` so the
loader can map registry id → class name. Without it the script falls
back to ``classes[id]`` and silently drops or mis-labels rows whose
registry id doesn't equal their classes.txt position.

Usage::

    python -m manual_reviewer.scripts.build_tasks_from_yolo \\
        --dataset /media/data_2/.../datatang_diverse_1000 \\
        --ls-url http://127.0.0.1:8080 \\
        --ls-project 9 \\
        --assignees pavan,sree,raj,sathish,deepak \\
        --assignment-strategy frame-count-rr
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote

from manual_reviewer.pipeline_io.clip_id import clip_prefix
from manual_reviewer.scripts.build_tasks import (
    _parse_assignees,
    _post_to_ls,
    _stable_assignee,
    assign_by_frame_count_rr,
)

logger = logging.getLogger("manual_reviewer.build_tasks_from_yolo")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dataset = args.dataset.resolve()
    classes_path = dataset / "classes.txt"
    if not classes_path.exists():
        # Fall back to yolo/classes.txt — the dataset_selection layout.
        classes_path = dataset / "yolo" / "classes.txt"
    labels_dir = args.labels_dir or dataset / "yolo" / "labels"
    stem_map_path = dataset / "stem_to_path.json"

    for required in (classes_path, labels_dir, stem_map_path):
        if not required.exists():
            logger.error("missing %s", required)
            return 2

    classes = _read_classes(classes_path)
    classes_set = set(classes)
    logger.info("classes (%d): %s", len(classes), classes[:5] + ["..."] if len(classes) > 5 else classes)

    id_to_name = _load_class_registry(args.aav4_config) if args.aav4_config else None
    if id_to_name:
        logger.info("class_registry: %d entries (mapping registry-id → class_name)",
                    len(id_to_name))
        unknown = [name for name in id_to_name.values() if name not in classes_set]
        if unknown:
            logger.warning(
                "registry classes missing from classes.txt (will be dropped): %s",
                sorted(set(unknown)),
            )
    else:
        logger.warning(
            "no --aav4-config; falling back to classes[id] which is wrong "
            "for aa_v4 YOLO exports — pass --aav4-config to fix"
        )

    stem_to_path = json.loads(stem_map_path.read_text(encoding="utf-8"))
    logger.info("stem_to_path entries: %d", len(stem_to_path))

    # Source the stems from the label-file set — that's the authoritative
    # "what's in this YOLO export" answer (1 stem = 1 .txt file).
    # `clips.txt` in the dataset_selection layout is clip *prefixes*,
    # not stems, and would have skipped most of the dataset.
    stems = sorted(p.stem for p in labels_dir.glob("*.txt"))
    if args.limit:
        stems = stems[:args.limit]
    logger.info("stems to import: %d", len(stems))

    try:
        assignees = _parse_assignees(args.assignees)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    if assignees:
        logger.info("assignment across %d users (strategy=%s): %s",
                    len(assignees), args.assignment_strategy, assignees)

    tasks: list[dict[str, Any]] = []
    skipped_no_image = 0
    skipped_no_label = 0

    for stem in stems:
        image_path = stem_to_path.get(stem)
        if not image_path:
            skipped_no_image += 1
            continue
        label_path = labels_dir / f"{stem}.txt"
        rows = _read_yolo_label(label_path) if label_path.exists() else []
        if label_path.exists() is False:
            skipped_no_label += 1
        task = _build_task(
            image_id=stem,
            image_path=image_path,
            yolo_rows=rows,
            classes=classes,
            id_to_name=id_to_name,
            image_url_template=args.image_url_template,
            model_version=args.model_version,
        )
        tasks.append(task)

    logger.info(
        "built %d tasks (skipped %d missing image_path, %d missing label file)",
        len(tasks), skipped_no_image, skipped_no_label,
    )
    if not tasks:
        logger.warning("nothing to import")
        return 1

    if assignees:
        if args.assignment_strategy == "hash":
            for t in tasks:
                t["data"]["assigned_to"] = _stable_assignee(
                    clip_prefix(t["data"]["image_id"]), assignees,
                )
        else:  # frame-count-rr
            clip_counts = Counter(clip_prefix(t["data"]["image_id"]) for t in tasks)
            clip_to_user = assign_by_frame_count_rr(dict(clip_counts), assignees)
            for t in tasks:
                t["data"]["assigned_to"] = clip_to_user[clip_prefix(t["data"]["image_id"])]
        counts = Counter(t["data"]["assigned_to"] for t in tasks)
        clip_counts_per_user: dict[str, set[str]] = {u: set() for u in assignees}
        for t in tasks:
            clip_counts_per_user[t["data"]["assigned_to"]].add(clip_prefix(t["data"]["image_id"]))
        logger.info("frames per user: %s",
                    ", ".join(f"{u}={counts.get(u, 0)}" for u in assignees))
        logger.info("clips per user:  %s",
                    ", ".join(f"{u}={len(clip_counts_per_user[u])}" for u in assignees))

    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
        logger.info("wrote %s", args.out_file)

    if args.ls_url:
        if not args.ls_token or args.ls_project is None:
            logger.error("--ls-url requires --ls-token and --ls-project")
            return 2
        posted = _post_to_ls(
            tasks,
            base_url=args.ls_url,
            token=args.ls_token,
            project_id=args.ls_project,
            skip_existing=args.skip_existing,
            timeout=args.ls_timeout,
            batch_size=args.ls_batch_size,
        )
        logger.info("posted %d tasks to LS project %s", posted, args.ls_project)

    return 0


def _read_classes(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_yolo_label(path: Path) -> list[tuple[int, float, float, float, float, float | None]]:
    """Parse YOLO rows: ``class_id cx cy w h [score]`` per line."""
    out: list[tuple[int, float, float, float, float, float | None]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 5:
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = (float(x) for x in parts[1:5])
            score = float(parts[5]) if len(parts) > 5 else None
        except ValueError:
            continue
        out.append((cls_id, cx, cy, w, h, score))
    return out


def _build_task(
    *,
    image_id: str,
    image_path: str,
    yolo_rows: list[tuple[int, float, float, float, float, float | None]],
    classes: list[str],
    id_to_name: dict[int, str] | None,
    image_url_template: str,
    model_version: str,
) -> dict[str, Any]:
    image_url = image_url_template.format(path=quote(image_path, safe="/"))
    classes_set = set(classes)
    regions: list[dict[str, Any]] = []
    for idx, (cls_id, cx, cy, w, h, score) in enumerate(yolo_rows):
        if id_to_name is not None:
            class_name = id_to_name.get(cls_id)
            if class_name is None or class_name not in classes_set:
                continue
        else:
            if not 0 <= cls_id < len(classes):
                continue
            class_name = classes[cls_id]
        x1 = max(0.0, cx - w / 2.0)
        y1 = max(0.0, cy - h / 2.0)
        x2 = min(1.0, cx + w / 2.0)
        y2 = min(1.0, cy + h / 2.0)
        if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
            continue
        cand_id = f"{image_id}_yolo{idx}"
        regions.append({
            "id": cand_id,
            "type": "rectanglelabels",
            "from_name": "bbox",
            "to_name": "image",
            "image_rotation": 0,
            "value": {
                "x": x1 * 100.0, "y": y1 * 100.0,
                "width": (x2 - x1) * 100.0, "height": (y2 - y1) * 100.0,
                "rotation": 0,
                "rectanglelabels": [class_name],
            },
            "meta": {"source_yolo_score": score},
        })
    predictions = (
        [{"model_version": model_version, "result": regions, "score": 1.0}]
        if regions else []
    )
    return {
        "data": {
            "image": image_url,
            "image_id": image_id,
            "image_path": image_path,
        },
        "predictions": predictions,
        "meta": {"image_id": image_id},
    }


def _load_class_registry(config_path: Path) -> dict[int, str]:
    """Read aa_v4 ``config.yaml`` and return ``{registry_id: class_name}``.

    aa_v4's YOLO exporter stamps ``class_registry[<name>].id`` into
    each label row, NOT the position in classes.txt — the two diverge
    when classes.txt is a packed subset of a larger registry.
    """
    import yaml  # noqa: WPS433 — lazy
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    reg = cfg.get("class_registry") or {}
    return {meta["id"]: name for name, meta in reg.items() if "id" in meta}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build LS tasks directly from a YOLO dataset")
    p.add_argument("--dataset", type=Path, required=True,
                   help="Dataset root containing classes.txt, yolo/labels/, "
                        "stem_to_path.json, clips.txt")
    p.add_argument("--labels-dir", type=Path, default=None,
                   help="Override label dir (default: <dataset>/yolo/labels)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--image-url-template",
                   default="/data/local-files/?d={path}")
    p.add_argument("--model-version", default="dataset_selection_yolo")
    p.add_argument("--out-file", type=Path, default=None,
                   help="Write tasks JSON for inspection (skips LS POST)")
    p.add_argument("--ls-url", default=None)
    p.add_argument("--ls-token",
                   default=os.environ.get("LS_TOKEN"))
    p.add_argument("--ls-project", type=int, default=None)
    p.add_argument("--ls-timeout", type=float, default=30.0)
    p.add_argument("--ls-batch-size", type=int, default=100)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--assignees", default=None,
                   help="Comma-separated reviewer names "
                        "(e.g. pavan,sree,raj,sathish,deepak)")
    p.add_argument("--assignment-strategy",
                   choices=["hash", "frame-count-rr"],
                   default="frame-count-rr")
    p.add_argument("--aav4-config", type=Path, default=None,
                   help="Path to the aa_v4 job's config.yaml — its "
                        "class_registry maps the YOLO file's class_id "
                        "to a class name. REQUIRED when the YOLO labels "
                        "were exported by aa_v4 (otherwise rows for "
                        "non-position-aligned classes get silently "
                        "mis-labeled or dropped).")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
