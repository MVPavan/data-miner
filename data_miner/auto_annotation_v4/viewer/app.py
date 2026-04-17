"""FastAPI backend for the auto_annotation_v4 viewer.

Replaces v3's file-based checkpoint reads with direct SQL queries against the
per-job ``pipeline.db`` SQLite database.  WAL journal mode allows concurrent
reads while the pipeline writes, so the viewer can run alongside a live
pipeline without contention.

Endpoints:
  GET /                       — Single-page frontend (index.html).
  GET /api/stems              — List all image_ids in the database.
  GET /api/data/{image_id}    — Full per-image data: stages, proposals, meta.
  GET /api/job                — Job-level info (config, classes, summary).
  GET /api/image/{image_id}   — Serve the source image file.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _is_within(path: Path, roots: list[Path]) -> bool:
    """True if *path* resolves inside any of *roots* (after symlink resolution).

    Used as a security check to prevent path traversal when serving images.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for r in roots:
        try:
            resolved.relative_to(r.resolve())
            return True
        except ValueError:
            continue
    return False


def create_app(job_dir: Path, image_dir: Path | None = None) -> FastAPI:
    """Create and return the FastAPI viewer application.

    All data is read from the SQLite database at ``{job_dir}/pipeline.db``
    using synchronous connections (SQLite WAL allows concurrent reads while
    the pipeline writes asynchronously).

    Parameters
    ----------
    job_dir:
        Root output directory for the pipeline job.  Must contain
        ``pipeline.db`` and optionally ``config.yaml``.
    image_dir:
        Optional additional directory to search for source images.
        Added to the allowed-roots list for path traversal protection.

    Returns
    -------
    FastAPI
        Configured application instance ready to be served.
    """
    job_dir = Path(job_dir)
    db_path = job_dir / "pipeline.db"

    # Image path allowlist — served files must resolve inside one of these.
    allowed_roots: list[Path] = [job_dir]
    if image_dir is not None:
        allowed_roots.append(Path(image_dir))

    # ------------------------------------------------------------------
    # SQLite helpers (sync reads via WAL — safe alongside async pipeline)
    # ------------------------------------------------------------------

    def _query(sql: str, params: tuple = ()) -> list[dict]:
        """Execute a read-only SQL query and return rows as dicts.

        Opens a fresh connection per request to avoid thread-safety issues
        with FastAPI's async workers.  WAL mode ensures reads never block
        the pipeline's async writes.
        """
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def _query_one(sql: str, params: tuple = ()) -> dict | None:
        """Execute a query and return the first row as a dict, or None."""
        rows = _query(sql, params)
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # Config / classes helpers
    # ------------------------------------------------------------------

    def _load_frozen_config() -> dict | None:
        """Load the frozen config.yaml (JSON) written by the pipeline."""
        cfg_path = job_dir / "config.yaml"
        if cfg_path.exists():
            try:
                return json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _load_classes() -> list[str]:
        """Load class names from classes.txt."""
        p = job_dir / "classes.txt"
        if not p.exists():
            return []
        return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def _load_class_id_map() -> dict[str, int]:
        """Build ``{class_name: class_id}`` from the frozen job config.

        Falls back to positional indexing of ``classes.txt`` when the
        config is missing (which may produce wrong ids with non-contiguous
        gaps).
        """
        data = _load_frozen_config()
        if isinstance(data, dict):
            reg = data.get("class_registry") or {}
            if isinstance(reg, dict):
                out: dict[str, int] = {}
                for name, cls_data in reg.items():
                    if isinstance(cls_data, dict) and "id" in cls_data:
                        out[name] = int(cls_data["id"])
                if out:
                    return out
        # Fallback — may produce wrong ids with gaps.
        return {name: i for i, name in enumerate(_load_classes())}

    def _load_summary() -> dict | None:
        """Load summary.json if it exists."""
        p = job_dir / "summary.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Image path resolution
    # ------------------------------------------------------------------

    def _find_image_path(image_id: str) -> Path | None:
        """Locate the source image for *image_id*.

        Resolution order:
          1. ``image_path`` from the ``image_meta`` table, only if it
             resolves inside one of the allowed roots.
          2. ``{image_id}.{ext}`` inside *image_dir*.
        """
        row = _query_one(
            "SELECT image_path FROM image_meta WHERE image_id = ?",
            (image_id,),
        )
        if row and row.get("image_path"):
            p = Path(row["image_path"])
            if p.exists() and _is_within(p, allowed_roots):
                return p

        if image_dir and Path(image_dir).exists():
            for ext in IMAGE_EXTS:
                p = Path(image_dir) / f"{image_id}{ext}"
                if p.exists() and _is_within(p, allowed_roots):
                    return p
        return None

    # ------------------------------------------------------------------
    # FastAPI app
    # ------------------------------------------------------------------

    app = FastAPI(title="AA v4 Viewer")

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        """Serve the single-page frontend."""
        index_path = static_dir / "index.html"
        if index_path.exists():
            return index_path.read_text(encoding="utf-8")
        return "<html><body><h1>AA v4 Viewer</h1><p>No index.html found in static/.</p></body></html>"

    @app.get("/api/job")
    async def get_job_info() -> dict[str, Any]:
        """Return job-level metadata, classes, and summary."""
        job_row = _query_one("SELECT * FROM job_info LIMIT 1") if db_path.exists() else None
        return {
            "job_dir": str(job_dir),
            "job_info": job_row,
            "classes": _load_classes(),
            "class_id_map": _load_class_id_map(),
            "summary": _load_summary(),
        }

    @app.get("/api/stems")
    async def get_stems() -> list[str]:
        """Return all image_ids ordered alphabetically."""
        if not db_path.exists():
            return []
        rows = _query("SELECT image_id FROM image_meta ORDER BY image_id")
        return [r["image_id"] for r in rows]

    @app.get("/api/data/{image_id}")
    async def get_data(image_id: str) -> dict[str, Any]:
        """Return full per-image data assembled from the SQLite database.

        Queries the ``image_meta``, ``stages``, and ``proposals`` tables
        and assembles a response with the same shape as v3 for frontend
        compatibility.
        """
        if not db_path.exists():
            raise HTTPException(404, "Database not found")

        # Image meta
        meta = _query_one(
            "SELECT * FROM image_meta WHERE image_id = ?", (image_id,)
        )
        if meta is None:
            raise HTTPException(404, f"No data for {image_id}")

        # Parse stages_completed from JSON string to list.
        if isinstance(meta.get("stages_completed"), str):
            try:
                meta["stages_completed"] = json.loads(meta["stages_completed"])
            except (json.JSONDecodeError, TypeError):
                meta["stages_completed"] = []

        # Stage checkpoints (detect, evaluate, refine, finalize).
        stages_data: dict[str, Any] = {}
        stage_rows = _query(
            "SELECT stage, data FROM stages WHERE image_id = ?", (image_id,)
        )
        for row in stage_rows:
            try:
                stages_data[row["stage"]] = json.loads(row["data"])
            except (json.JSONDecodeError, TypeError):
                stages_data[row["stage"]] = row["data"]

        # Per-model proposals.
        proposals: dict[str, Any] = {}
        proposal_rows = _query(
            "SELECT model, data FROM proposals WHERE image_id = ?", (image_id,)
        )
        for row in proposal_rows:
            try:
                proposals[row["model"]] = json.loads(row["data"])
            except (json.JSONDecodeError, TypeError):
                proposals[row["model"]] = row["data"]

        # Trace and review files (still file-based from OutputWriter).
        trace_path = job_dir / "traces" / f"{image_id}.json"
        trace = None
        if trace_path.exists():
            try:
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        review_path = job_dir / "review" / f"{image_id}.json"
        review = None
        if review_path.exists():
            try:
                review = json.loads(review_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        # YOLO labels (still file-based from OutputWriter).
        labels_path = job_dir / "labels" / f"{image_id}.txt"
        labels = ""
        if labels_path.exists():
            try:
                labels = labels_path.read_text(encoding="utf-8")
            except Exception:
                pass

        # Image dimensions: prefer detect stage's image_size, then probe file.
        img_w, img_h = 0, 0
        detect_data = stages_data.get("detect")
        if isinstance(detect_data, dict):
            img_size = detect_data.get("image_size")
            if isinstance(img_size, list) and len(img_size) == 2:
                img_w, img_h = int(img_size[0]), int(img_size[1])

        img_path = _find_image_path(image_id)
        if img_path is not None and (img_w == 0 or img_h == 0):
            try:
                from PIL import Image

                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception:
                pass

        # Failures for this image (if any).
        failures = _query(
            "SELECT stage, attempts, last_error FROM failures WHERE image_id = ?",
            (image_id,),
        )

        data: dict[str, Any] = {
            "image_id": image_id,
            "meta": meta,
            "detect": stages_data.get("detect"),
            "evaluate": stages_data.get("evaluate"),
            "refine": stages_data.get("refine"),
            "finalize": stages_data.get("finalize"),
            "proposals": proposals,
            "trace": trace,
            "review": review,
            "labels": labels,
            "classes": _load_classes(),
            "class_id_map": _load_class_id_map(),
            "image_width": img_w,
            "image_height": img_h,
            "image_url": f"/api/image/{image_id}" if img_path else None,
            "failures": failures if failures else None,
        }

        return data

    @app.get("/api/image/{image_id}")
    async def get_image(image_id: str):
        """Serve the source image file for *image_id*.

        The image path is resolved from the ``image_meta`` table or by
        scanning ``image_dir``.  Only files within the allowed roots are
        served to prevent path traversal.
        """
        img_path = _find_image_path(image_id)
        if not img_path or not img_path.exists():
            raise HTTPException(404, f"Image not found for {image_id}")
        media_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }.get(img_path.suffix.lower(), "image/jpeg")
        return FileResponse(img_path, media_type=media_type)

    return app
