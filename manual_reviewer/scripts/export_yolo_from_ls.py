"""Export a clean YOLO dataset directly from Label Studio.

No pipeline.db round-trip. Pulls tasks + annotations via the LS REST API
(or a local LS-export JSON file), filters by frame_state and annotator,
applies a defensive last-wins fix for the rectanglelabels-append bug,
and writes a standard Ultralytics-shaped dataset:

    out_dir/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── manifest.csv

Usage (live LS):
    python -m manual_reviewer.scripts.export_yolo_from_ls \\
        --ls-url http://localhost:8080 --ls-token <token> --ls-project 9 \\
        --image-root /media/data_2/datasets/datasets_pavan/DataTang_val \\
        --classes-file output/auto_annotation_v4/datatang_val_detect/classes.txt \\
        --out-dir export/datatang_yolo \\
        --frame-state clean \\
        --reviewer pavan,sandeep \\
        --val-split 0.1

Offline workflow (snapshot once, filter many times):

    # 1. snapshot LS once (repeat any time you want fresh data)
    curl -s -H "Authorization: Token $LS_TOKEN" \\
      "http://localhost:8080/api/projects/9/export?exportType=JSON&download_all_tasks=true" \\
      > /path/snapshots/ls_project9_$(date +%F).json
    curl -s -H "Authorization: Token $LS_TOKEN" \\
      "http://localhost:8080/api/users/" \\
      > /path/snapshots/ls_users_$(date +%F).json

    # 2. filter from disk; iterate filters as much as you want
    python -m manual_reviewer.scripts.export_yolo_from_ls \\
        --in-file /path/snapshots/ls_project9_2026-05-06.json \\
        --users-file /path/snapshots/ls_users_2026-05-06.json \\
        --image-root /media/data_2/datasets/datasets_pavan/DataTang_val \\
        --classes-file output/auto_annotation_v4/datatang_val_detect/classes.txt \\
        --out-dir export/pavan_clean_yolo \\
        --frame-state clean \\
        --reviewer pavan

Filters (all defaults are conservative — clean only, no skipped frames):
  --frame-state          comma-separated; default: "clean"
                         choices: clean, needs_more_review, ambiguous_skip
  --reviewer             comma-separated names (matches LS user.first_name
                         or .email's local-part); default: any reviewer
  --include-cancelled    include was_cancelled=True annotations; default: skip
  --include-deletions    keep regions the reviewer deleted (rare)

Output behaviour:
  --copy-images          copy source jpgs (default: symlink)
  --val-split 0.1        deterministic per-image_id hash split
  --classes-file PATH    one class name per line; line N -> class id N
                         (matches output/auto_annotation_v4/.../classes.txt)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger("export_yolo_from_ls")


# ---------------------------------------------------------------------------
# Class registry
# ---------------------------------------------------------------------------


def load_class_index(classes_file: Path) -> dict[str, int]:
    """{class_name → class_id} by line position. Class IDs are 0-indexed."""
    names = [
        ln.strip() for ln in classes_file.read_text().splitlines() if ln.strip()
    ]
    return {n: i for i, n in enumerate(names)}


# ---------------------------------------------------------------------------
# LS REST
# ---------------------------------------------------------------------------


def fetch_users(ls_url: str, token: str) -> dict[int, dict[str, Any]]:
    r = requests.get(f"{ls_url.rstrip('/')}/api/users/",
                     headers={"Authorization": f"Token {token}"}, timeout=30)
    r.raise_for_status()
    return {u["id"]: u for u in r.json() if isinstance(u, dict)}


def fetch_tasks_with_annotations(
    ls_url: str, token: str, project_id: int,
) -> list[dict[str, Any]]:
    """Use LS export endpoint to pull tasks+annotations in one shot."""
    url = f"{ls_url.rstrip('/')}/api/projects/{project_id}/export"
    params = {"exportType": "JSON", "download_all_tasks": "true"}
    r = requests.get(url, params=params,
                     headers={"Authorization": f"Token {token}"}, timeout=600)
    r.raise_for_status()
    return r.json()


def load_in_file(path: Path) -> list[dict[str, Any]]:
    """Read an LS export JSON file (alternative to live API)."""
    body = json.loads(path.read_text())
    if isinstance(body, dict):
        body = body.get("tasks") or body.get("data") or []
    if not isinstance(body, list):
        raise ValueError(f"unexpected shape in {path}")
    return body


# ---------------------------------------------------------------------------
# Reviewer name resolution
# ---------------------------------------------------------------------------


def reviewer_name(user_obj: dict[str, Any] | None, completed_by: Any) -> str:
    """Best-effort name string for filtering/manifest.

    Priority: first_name → email local-part → "user_<id>" → "unknown".
    """
    if user_obj:
        fn = (user_obj.get("first_name") or "").strip()
        if fn:
            return fn.lower()
        em = (user_obj.get("email") or "").strip()
        if em:
            return em.split("@", 1)[0].lower()
        if user_obj.get("id") is not None:
            return f"user_{user_obj['id']}"
    if isinstance(completed_by, dict):
        return reviewer_name(completed_by, None)
    if completed_by is not None:
        return f"user_{completed_by}"
    return "unknown"


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------


def extract_frame_state(results: list[dict[str, Any]]) -> str:
    for r in results:
        if (r.get("type") in ("choices", "select")
                and r.get("from_name") == "frame_state"):
            choices = (r.get("value") or {}).get("choices") or []
            if choices:
                return str(choices[0])
    return "clean"  # XML default


def yolo_line_from_region(
    region: dict[str, Any],
    class_index: dict[str, int],
) -> str | None:
    """Convert one LS rectanglelabels region into a YOLO label line.

    Applies the *last-wins* rule on `rectanglelabels` to defend against the
    LS bug that appends class names when a reviewer changes class on a box.
    """
    if region.get("type") != "rectanglelabels":
        return None
    val = region.get("value") or {}
    labels = val.get("rectanglelabels") or []
    if not labels:
        return None
    cls_name = labels[-1]  # bug-fix: last-wins
    if cls_name not in class_index:
        logger.debug("skipping unknown class %r", cls_name)
        return None
    cid = class_index[cls_name]

    # LS rectangle values come back as percentages; normalize to [0,1].
    try:
        x = float(val["x"]) / 100.0
        y = float(val["y"]) / 100.0
        w = float(val["width"]) / 100.0
        h = float(val["height"]) / 100.0
    except (KeyError, TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    cx = max(0.0, min(1.0, x + w / 2.0))
    cy = max(0.0, min(1.0, y + h / 2.0))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


# ---------------------------------------------------------------------------
# Dataset writer
# ---------------------------------------------------------------------------


def split_for(image_id: str, val_split: float) -> str:
    if val_split <= 0:
        return "train"
    h = int(hashlib.md5(image_id.encode()).hexdigest(), 16) % 1000
    return "val" if h < int(val_split * 1000) else "train"


def place_image(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def write_data_yaml(out_dir: Path, class_index: dict[str, int]) -> None:
    inv = {i: n for n, i in class_index.items()}
    names = [inv[i] for i in sorted(inv)]
    lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--in-file", type=Path,
                     help="LS export JSON file (alternative to API). "
                          "Pair with --users-file for offline reviewer-name resolution.")
    src.add_argument("--ls-project", type=int,
                     help="LS project id; requires --ls-url and --ls-token")
    p.add_argument("--ls-url", default="http://localhost:8080")
    p.add_argument("--ls-token")
    p.add_argument("--users-file", type=Path,
                   help="JSON dump of /api/users/; used with --in-file for "
                        "offline reviewer-name resolution.")
    p.add_argument("--image-root", required=True, type=Path,
                   help="Where source jpgs live; used to resolve image_path")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--classes-file", required=True, type=Path)
    p.add_argument("--frame-state", default="clean",
                   help="comma-separated; choices clean,needs_more_review,ambiguous_skip")
    p.add_argument("--reviewer", default="",
                   help="comma-separated reviewer names (first_name or email local-part); default: any")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--copy-images", action="store_true",
                   help="copy jpgs (default: symlink)")
    p.add_argument("--include-cancelled", action="store_true")
    p.add_argument("--include-deletions", action="store_true",
                   help="keep regions reviewer deleted (rare; mostly for diff debugging)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    class_index = load_class_index(args.classes_file)
    if not class_index:
        logger.error("classes file %s empty/missing", args.classes_file)
        return 2
    logger.info("loaded %d classes from %s", len(class_index), args.classes_file)

    frame_states = {s.strip() for s in args.frame_state.split(",") if s.strip()}
    reviewer_filter = {r.strip().lower() for r in args.reviewer.split(",") if r.strip()}

    # 1. Fetch tasks
    if args.in_file:
        tasks = load_in_file(args.in_file)
        users_by_id: dict[int, dict[str, Any]] = {}
        if args.users_file:
            users_raw = json.loads(args.users_file.read_text())
            users_by_id = {u["id"]: u for u in users_raw if isinstance(u, dict)}
            logger.info("loaded %d users from %s", len(users_by_id), args.users_file)
        logger.info("loaded %d tasks from %s", len(tasks), args.in_file)
    else:
        if not args.ls_token:
            logger.error("--ls-token required when --ls-project is set")
            return 2
        users_by_id = fetch_users(args.ls_url, args.ls_token)
        tasks = fetch_tasks_with_annotations(args.ls_url, args.ls_token, args.ls_project)
        logger.info("fetched %d tasks, %d users from LS project %d",
                    len(tasks), len(users_by_id), args.ls_project)

    # 2. Walk completions, build (image_id, label_lines, meta) records
    skipped: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for task in tasks:
        data = task.get("data") or {}
        image_id = data.get("image_id")
        image_path = data.get("image_path")
        if not image_id or not image_path:
            skipped["no_image_meta"] += 1
            continue
        annotations = task.get("annotations") or task.get("completions") or []
        # Pick latest non-cancelled annotation (LS allows multiple drafts).
        valid = [
            a for a in annotations
            if isinstance(a, dict) and (
                args.include_cancelled or not a.get("was_cancelled")
            )
        ]
        if not valid:
            skipped["no_annotation"] += 1
            continue
        ann = max(valid, key=lambda a: a.get("updated_at") or a.get("created_at") or "")

        # Filter by frame_state
        results = ann.get("result") or []
        fs = extract_frame_state(results)
        if fs not in frame_states:
            skipped[f"frame_state={fs}"] += 1
            continue

        # Resolve reviewer
        completed_by = ann.get("completed_by")
        uid = (completed_by.get("id") if isinstance(completed_by, dict)
               else completed_by)
        user_obj = users_by_id.get(uid) if isinstance(uid, int) else None
        rname = reviewer_name(user_obj, completed_by)
        if reviewer_filter and rname not in reviewer_filter:
            skipped[f"reviewer={rname}"] += 1
            continue

        # Apply deletions filter (LS marks deleted by region absence already;
        # honour --include-deletions only when explicit `is_deleted` flags
        # appear, which is rare).
        regions = [
            r for r in results
            if isinstance(r, dict) and r.get("type") == "rectanglelabels"
            and (args.include_deletions or not r.get("is_deleted"))
        ]
        lines: list[str] = []
        per_class: Counter[str] = Counter()
        for region in regions:
            yl = yolo_line_from_region(region, class_index)
            if yl is None:
                continue
            lines.append(yl)
            cid = int(yl.split(" ", 1)[0])
            inv = {i: n for n, i in class_index.items()}
            per_class[inv[cid]] += 1
        if not lines:
            skipped["no_valid_boxes"] += 1
            continue

        records.append({
            "image_id": image_id,
            "image_path": image_path,
            "frame_state": fs,
            "reviewer": rname,
            "lines": lines,
            "per_class": per_class,
            "n_boxes": len(lines),
            "annotation_id": ann.get("id"),
            "ann_updated_at": ann.get("updated_at"),
        })

    if not records:
        logger.error("no tasks survived filtering. skipped breakdown: %s",
                     dict(skipped))
        return 1
    logger.info("%d images survived filtering. Skipped breakdown: %s",
                len(records), dict(skipped))

    # 3. Write dataset
    if args.dry_run:
        # Show summary only
        cls_total: Counter[str] = Counter()
        rev_total: Counter[str] = Counter()
        split_total: Counter[str] = Counter()
        for r in records:
            cls_total.update(r["per_class"])
            rev_total[r["reviewer"]] += 1
            split_total[split_for(r["image_id"], args.val_split)] += 1
        logger.info("DRY RUN — would write: train=%d val=%d", split_total["train"], split_total["val"])
        logger.info("by reviewer: %s", dict(rev_total))
        logger.info("by class: %s", dict(cls_total))
        return 0

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    written: Counter[str] = Counter()
    missing_src: list[str] = []
    rows: list[list[Any]] = []
    for rec in records:
        split = split_for(rec["image_id"], args.val_split)
        # Resolve source jpg
        src = Path(rec["image_path"])
        if not src.is_absolute():
            src = args.image_root / src
        if not src.exists():
            # Try resolving by stem under image_root
            fallback = args.image_root / Path(rec["image_path"]).name
            if fallback.exists():
                src = fallback
            else:
                missing_src.append(rec["image_id"])
                continue

        stem = rec["image_id"]
        img_dst = out_dir / "images" / split / f"{stem}{src.suffix}"
        lbl_dst = out_dir / "labels" / split / f"{stem}.txt"
        place_image(src, img_dst, args.copy_images)
        lbl_dst.write_text("\n".join(rec["lines"]) + "\n")
        written[split] += 1
        rows.append([
            rec["image_id"], split, rec["frame_state"], rec["reviewer"],
            rec["n_boxes"], "|".join(f"{k}:{v}" for k, v in rec["per_class"].items()),
            rec["annotation_id"], rec["ann_updated_at"],
        ])

    # 4. Manifest + data.yaml
    with (out_dir / "manifest.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "split", "frame_state", "reviewer",
                    "n_boxes", "per_class", "annotation_id", "ann_updated_at"])
        w.writerows(rows)
    write_data_yaml(out_dir, class_index)

    logger.info("done. train=%d val=%d → %s",
                written["train"], written["val"], out_dir)
    if missing_src:
        logger.warning("%d images had no source jpg under %s; first 5: %s",
                       len(missing_src), args.image_root, missing_src[:5])
    return 0


if __name__ == "__main__":
    sys.exit(main())
