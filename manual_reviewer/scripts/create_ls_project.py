"""Bootstrap a Label Studio project from a dataset's ``classes.txt``.

One command does everything you need before importing tasks:

  1. Render the labeling XML from ``classes.txt`` (24-class palette,
     hotkeys 1-0/q-w-e-r-t-y-u-i-o-p/a-s-d, "v" reserved for V-tool).
  2. ``POST /api/projects/`` with the rendered XML as ``label_config``.
  3. ``POST /api/storages/localfiles`` to expose dataset images at
     ``/data/local-files/?d=<image_path>``.
  4. ``POST /api/ml/`` to connect the ml_backend (interactive smart tools).
  5. Print the new project_id so the caller can pipe it into
     ``build_tasks.py --ls-project <id>``.

Each step is independently failure-isolated and re-runnable: if step 4
fails (e.g., the ml_backend is offline), the project still exists with
storage attached, and you can re-run with ``--project-id`` to pick up
where it left off.

Usage::

    python -m manual_reviewer.scripts.create_ls_project \\
        --classes /path/to/classes.txt \\
        --dataset-path /path/to/images \\
        --title "my_review_v1" \\
        --ls-url http://localhost:8080 \\
        --ls-token "$LS_TOKEN" \\
        --ml-backend-url http://127.0.0.1:9090

The token can also flow in via the ``LS_TOKEN`` env var (preferred — keeps
it out of ``ps auxf``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from manual_reviewer.configs.build_labeling_config import render as render_xml

logger = logging.getLogger("manual_reviewer.create_ls_project")


def _http_post(url: str, headers: dict[str, str], body: dict[str, Any], *, timeout: float):
    import httpx

    with httpx.Client(timeout=timeout, headers=headers) as client:
        resp = client.post(url, json=body)
    return resp


def _http_patch(url: str, headers: dict[str, str], body: dict[str, Any], *, timeout: float):
    import httpx

    with httpx.Client(timeout=timeout, headers=headers) as client:
        resp = client.patch(url, json=body)
    return resp


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    token = args.ls_token or os.environ.get("LS_TOKEN")
    if not token:
        logger.error("--ls-token or LS_TOKEN env var required")
        return 2

    classes_path = args.classes.resolve()
    if not classes_path.exists():
        logger.error("classes file not found: %s", classes_path)
        return 2

    # Step 1 — render XML from classes.txt
    xml = render_xml(classes_path)
    if args.write_xml_to:
        args.write_xml_to.parent.mkdir(parents=True, exist_ok=True)
        args.write_xml_to.write_text(xml, encoding="utf-8")
        logger.info("wrote rendered XML → %s", args.write_xml_to)

    base_url = args.ls_url.rstrip("/")
    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}

    # Step 2 — create project (or update existing)
    if args.project_id is not None:
        resp = _http_patch(
            f"{base_url}/api/projects/{args.project_id}",
            headers,
            {"label_config": xml,
             **({"title": args.title} if args.title else {}),
             **({"description": args.description} if args.description else {})},
            timeout=args.timeout,
        )
        if resp.status_code >= 300:
            logger.error("PATCH project %s failed: HTTP %s — %s",
                         args.project_id, resp.status_code, resp.text[:300])
            return 3
        project_id = args.project_id
        logger.info("PATCHed existing project %s with rendered XML", project_id)
    else:
        body: dict[str, Any] = {
            "label_config": xml,
            "maximum_annotations": 1,
            "show_annotation_history": False,
            "enable_empty_annotation": True,
        }
        if args.title:
            body["title"] = args.title
        if args.description:
            body["description"] = args.description
        resp = _http_post(f"{base_url}/api/projects/", headers, body, timeout=args.timeout)
        if resp.status_code >= 300:
            logger.error("POST /api/projects failed: HTTP %s — %s",
                         resp.status_code, resp.text[:300])
            return 3
        project_id = resp.json().get("id")
        logger.info("created project id=%s title=%r", project_id, body.get("title"))

    # Step 3 — attach Local Files storage (idempotency: LS allows duplicates;
    # if you re-run this command, you'll get a second storage entry. The
    # caller can pass --skip-storage when re-invoking.)
    if not args.skip_storage:
        resp = _http_post(
            f"{base_url}/api/storages/localfiles",
            headers,
            {
                "project": project_id,
                "title": args.dataset_path.name or "dataset",
                "path": str(args.dataset_path.resolve()),
                "use_blob_urls": False,
            },
            timeout=args.timeout,
        )
        if resp.status_code >= 300:
            logger.error("POST /api/storages/localfiles failed: HTTP %s — %s",
                         resp.status_code, resp.text[:300])
            # Non-fatal — keep going so the caller can re-attach manually.
        else:
            sid = resp.json().get("id")
            logger.info("attached Local Files storage id=%s path=%s", sid, args.dataset_path)

    # Step 4 — connect ML backend (interactive smart tools)
    if args.ml_backend_url and not args.skip_ml_backend:
        resp = _http_post(
            f"{base_url}/api/ml/",
            headers,
            {
                "project": project_id,
                "url": args.ml_backend_url,
                "title": "manual_reviewer ml_backend",
                "is_interactive": True,
            },
            timeout=args.timeout,
        )
        if resp.status_code >= 300:
            logger.error("POST /api/ml/ failed: HTTP %s — %s",
                         resp.status_code, resp.text[:300])
        else:
            mid = resp.json().get("id")
            logger.info("connected ml_backend id=%s url=%s", mid, args.ml_backend_url)

    # Step 5 — register annotation backup webhook (push-based on-disk
    # capture; lossless audit trail). Independent of pipeline.db sync.
    if args.enable_annotation_webhook:
        webhook_url = (
            args.annotation_webhook_url
            or (args.ml_backend_url + "/lswebhook/annotations"
                if args.ml_backend_url else None)
        )
        if webhook_url is None:
            logger.error(
                "--enable-annotation-webhook requires --ml-backend-url "
                "(or --annotation-webhook-url) to derive the target URL"
            )
        else:
            resp = _http_post(
                f"{base_url}/api/webhooks/",
                headers,
                {
                    "project": project_id,
                    "url": webhook_url,
                    "send_payload": True,
                    "send_for_all_actions": False,
                    # LS 1.23 action names — note delete is always plural.
                    "actions": [
                        "ANNOTATION_CREATED",
                        "ANNOTATION_UPDATED",
                        "ANNOTATIONS_CREATED",
                        "ANNOTATIONS_DELETED",
                    ],
                    "is_active": True,
                    "headers": {},
                },
                timeout=args.timeout,
            )
            if resp.status_code >= 300:
                logger.error("POST /api/webhooks/ failed: HTTP %s — %s",
                             resp.status_code, resp.text[:300])
            else:
                wid = resp.json().get("id")
                logger.info(
                    "registered annotation webhook id=%s url=%s",
                    wid, webhook_url,
                )

    # Print just the project_id to stdout so callers can pipe it.
    print(json.dumps({"project_id": project_id, "ls_url": base_url}))
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--classes", required=True, type=Path,
                   help="classes.txt — one class per line")
    p.add_argument("--dataset-path", required=True, type=Path,
                   help="filesystem dir served by LS Local Files storage")
    p.add_argument("--ls-url", default="http://localhost:8080",
                   help="Label Studio base URL")
    p.add_argument("--ls-token", default=None,
                   help="LS API token; falls back to LS_TOKEN env var")
    p.add_argument("--ml-backend-url", default=None,
                   help="ml_backend URL for interactive smart tools "
                        "(e.g. http://127.0.0.1:9090)")
    p.add_argument("--title", default=None, help="LS project title")
    p.add_argument("--description", default=None, help="LS project description")
    p.add_argument("--project-id", default=None, type=int,
                   help="if set, PATCH existing project with new XML "
                        "instead of creating a new one")
    p.add_argument("--write-xml-to", default=None, type=Path,
                   help="also persist the rendered XML to this path "
                        "(e.g. manual_reviewer/configs/labeling_config.xml)")
    p.add_argument("--skip-storage", action="store_true",
                   help="don't attach Local Files storage on this run "
                        "(useful when re-running with --project-id)")
    p.add_argument("--skip-ml-backend", action="store_true",
                   help="don't connect ML backend on this run")
    p.add_argument("--enable-annotation-webhook", action="store_true",
                   help="register an LS webhook that pushes annotation "
                        "events (CREATED/UPDATED/DELETED) to ml_backend's "
                        "/lswebhook/annotations route. NOTE: in our LS "
                        "Community 1.23 install this fires unreliably "
                        "when an ML backend is also connected. Prefer "
                        "the cron-driven `sync_ls_to_disk.py` for now; "
                        "leave this flag off until LS-side is fixed.")
    p.add_argument("--annotation-webhook-url", default=None,
                   help="explicit webhook URL override; default is "
                        "<ml-backend-url>/lswebhook/annotations")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="HTTP timeout in seconds (default 30)")
    return p.parse_args(argv)


if __name__ == "__main__":
    sys.exit(main())
