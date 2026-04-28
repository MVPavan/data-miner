"""FastAPI backend for the auto_annotation_v4 viewer.

Replaces v3's file-based checkpoint reads with direct SQL queries against the
per-job ``pipeline.db`` SQLite database.  WAL journal mode allows concurrent
reads while the pipeline writes, so the viewer can run alongside a live
pipeline without contention.

Endpoints:
  GET /                       — Single-page frontend (index.html).
  GET /api/stems              — Paginated list of image_ids with status/stages.
                                Query params: offset, limit, q, status, stage.
  GET /api/data/{image_id}    — Full per-image data: stages, proposals, meta.
  GET /api/job                — Job-level info (config, classes, summary).
  GET /api/image/{image_id}   — Serve the source image file.
  GET /api/search/schema      — Available classes/statuses/reasons per stage.
  GET /api/search             — Stage-aware class/status search → image list.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# Canonical pipeline stage order; used for the sidebar "stage completed" filter.
# ``human_review`` is event-driven (written by manual_reviewer/scripts/export_to_aa_v4.py)
# rather than worker-driven, but it shows up in image_meta.stages_completed so the
# viewer's filter UI must offer it as a selectable stage.
PIPELINE_STAGES = ("detect", "filter", "evaluate", "refine", "finalize", "human_review")

# Static schema for /api/search/schema. Class lists are populated from each
# stage's lazy-built index when present, falling back to classes.txt.
_FILTER_DROP_REASONS = (
    "source_model", "dedup", "geometric_filter", "head_without_person",
    "per_class_cap", "cross_class", "class_agnostic_nms", "score_floor",
    "rejected_upstream",
)
_DETECT_SOURCE_MODELS = ("sam3_dart", "grounding_dino", "sam3", "falcon", "owlv2")

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

    # Auto-add job_info.image_dir (recorded when the job was submitted) so
    # the viewer can serve source images without a CLI --image-dir override.
    # This is the submission-time directory and is trusted by construction.
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
            try:
                row = conn.execute(
                    "SELECT image_dir FROM job_info LIMIT 1"
                ).fetchone()
            finally:
                conn.close()
            if row and row[0]:
                recorded = Path(row[0])
                if recorded.exists() and recorded not in allowed_roots:
                    allowed_roots.append(recorded)
        except sqlite3.Error:
            pass

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
            conn.execute("PRAGMA query_only = TRUE")
            conn.execute("PRAGMA cache_size = -32768")
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
            "job_id": job_dir.name,
            "job_info": job_row,
            "classes": _load_classes(),
            "class_id_map": _load_class_id_map(),
            "summary": _load_summary(),
        }

    # ------------------------------------------------------------------
    # Stage-aware class search
    # ------------------------------------------------------------------
    # Lazy in-memory inverted indices, keyed by stage name. First request
    # for a stage scans all rows in `stages` for that stage and parses
    # the JSON; subsequent queries hit the cache.
    _class_index_cache: dict[str, dict] = {}

    def _orig_class_from_cid(cid: str) -> str | None:
        """Parse '<source_model>:<class>:<idx>' candidate ID → original class."""
        if not cid:
            return None
        parts = cid.split(":", 2)
        return parts[1] if len(parts) >= 2 else None

    def _build_index_detect() -> dict[str, Any]:
        rows = _query("SELECT image_id, data FROM stages WHERE stage='detect'")
        by_class: dict[str, set[str]] = {}
        by_class_model: dict[tuple[str, str], set[str]] = {}
        classes: set[str] = set()
        models: set[str] = set()
        for r in rows:
            try:
                d = json.loads(r["data"])
            except (json.JSONDecodeError, TypeError):
                continue
            img = r["image_id"]
            for c in d.get("candidates") or []:
                cn = c.get("class_name")
                sm = c.get("source_model")
                if cn:
                    classes.add(cn)
                    by_class.setdefault(cn, set()).add(img)
                if cn and sm:
                    models.add(sm)
                    by_class_model.setdefault((cn, sm), set()).add(img)
        return {
            "by_class": by_class,
            "by_class_model": by_class_model,
            "classes": sorted(classes),
            "source_models": sorted(models),
        }

    def _build_index_filter() -> dict[str, Any]:
        rows = _query("SELECT image_id, data FROM stages WHERE stage='filter'")
        auto_acc: dict[str, set[str]] = {}
        needs_eval: dict[str, set[str]] = {}
        dropped: dict[tuple[str, str], set[str]] = {}  # (reason, class) → ids
        classes: set[str] = set()
        reasons: set[str] = set()
        for r in rows:
            try:
                d = json.loads(r["data"])
            except (json.JSONDecodeError, TypeError):
                continue
            img = r["image_id"]
            cands_by_id = {
                c.get("candidate_id"): c.get("class_name")
                for c in d.get("candidates") or []
                if c.get("candidate_id")
            }
            routing = d.get("routing") or {}
            for cid in routing.get("auto_accepted") or []:
                cn = cands_by_id.get(cid) or _orig_class_from_cid(cid)
                if cn:
                    classes.add(cn)
                    auto_acc.setdefault(cn, set()).add(img)
            for cid in routing.get("needs_evaluation") or []:
                cn = cands_by_id.get(cid) or _orig_class_from_cid(cid)
                if cn:
                    classes.add(cn)
                    needs_eval.setdefault(cn, set()).add(img)
            for dr in d.get("drops") or []:
                reason = dr.get("reason") or "?"
                cid = dr.get("candidate_id") or ""
                cn = _orig_class_from_cid(cid)
                if cn:
                    classes.add(cn)
                    reasons.add(reason)
                    dropped.setdefault((reason, cn), set()).add(img)
        return {
            "auto_accepted": auto_acc,
            "needs_evaluation": needs_eval,
            "dropped": dropped,
            "classes": sorted(classes),
            "reasons": sorted(reasons),
        }

    def _build_index_evaluate() -> dict[str, Any]:
        rows = _query("SELECT image_id, data FROM stages WHERE stage='evaluate'")
        by_status_orig: dict[tuple[str, str], set[str]] = {}
        by_status_verdict: dict[tuple[str, str], set[str]] = {}
        relabel_pairs: dict[tuple[str, str], set[str]] = {}
        classes_orig: set[str] = set()
        classes_verdict: set[str] = set()
        for r in rows:
            try:
                d = json.loads(r["data"])
            except (json.JSONDecodeError, TypeError):
                continue
            img = r["image_id"]
            relabels_dict = d.get("relabels") or {}
            cid_status: dict[str, str] = {}
            for status_name in ("accepted", "review", "rejected", "drops"):
                for entry in d.get(status_name) or []:
                    # accepted/review/rejected are lists of cid strings;
                    # drops is a list of {candidate_id, reason, ...} dicts.
                    cid = entry.get("candidate_id") if isinstance(entry, dict) else entry
                    if isinstance(cid, str):
                        cid_status[cid] = status_name
            for v in d.get("verdicts") or []:
                cid = v.get("candidate_id") or ""
                orig = _orig_class_from_cid(cid)
                verdict = (
                    v.get("correct_class")
                    or relabels_dict.get(cid)
                    or v.get("detected_class")
                    or orig
                )
                status = cid_status.get(cid, "drops")
                if orig:
                    classes_orig.add(orig)
                    by_status_orig.setdefault((status, orig), set()).add(img)
                if verdict:
                    classes_verdict.add(verdict)
                    by_status_verdict.setdefault((status, verdict), set()).add(img)
                if orig and verdict and orig != verdict:
                    by_status_orig.setdefault(("relabeled", orig), set()).add(img)
                    by_status_verdict.setdefault(("relabeled", verdict), set()).add(img)
                    relabel_pairs.setdefault((orig, verdict), set()).add(img)
        return {
            "by_status_original": by_status_orig,
            "by_status_verdict": by_status_verdict,
            "relabel_pairs": relabel_pairs,
            "classes_original": sorted(classes_orig),
            "classes_verdict": sorted(classes_verdict),
            "statuses": ["accepted", "review", "rejected", "drops", "relabeled"],
        }

    def _build_index_finalize() -> dict[str, Any]:
        rows = _query("SELECT image_id, data FROM stages WHERE stage='finalize'")
        accepted: dict[str, set[str]] = {}
        review: dict[str, set[str]] = {}
        dropped: dict[str, set[str]] = {}
        classes: set[str] = set()
        for r in rows:
            try:
                d = json.loads(r["data"])
            except (json.JSONDecodeError, TypeError):
                continue
            img = r["image_id"]
            for a in d.get("final_annotations") or []:
                cn = a.get("class_name")
                if cn:
                    classes.add(cn)
                    accepted.setdefault(cn, set()).add(img)
            for a in d.get("review_items") or []:
                cn = a.get("class_name")
                if cn:
                    classes.add(cn)
                    review.setdefault(cn, set()).add(img)
            for a in d.get("dropped") or []:
                cn = a.get("class_name") or _orig_class_from_cid(a.get("candidate_id") or "")
                if cn:
                    classes.add(cn)
                    dropped.setdefault(cn, set()).add(img)
        return {
            "accepted": accepted,
            "review": review,
            "dropped": dropped,
            "classes": sorted(classes),
        }

    def _build_index_refine() -> dict[str, Any]:
        # Refine isn't run on current jobs — return empty scaffold so the UI
        # can still show the option for future jobs that do run refine.
        return {"classes": [], "statuses": []}

    _BUILDERS = {
        "detect": _build_index_detect,
        "filter": _build_index_filter,
        "evaluate": _build_index_evaluate,
        "refine": _build_index_refine,
        "finalize": _build_index_finalize,
    }

    def _get_index(stage: str) -> dict:
        if stage not in _class_index_cache:
            _class_index_cache[stage] = _BUILDERS[stage]()
        return _class_index_cache[stage]

    def _csv(s: str) -> list[str]:
        return [p.strip() for p in (s or "").split(",") if p.strip()]

    def _resolve_search(
        stage: str, classes: list[str], statuses: list[str], reasons: list[str],
        class_role: str, relabel_from: str, relabel_to: str, source_models: list[str],
    ) -> set[str]:
        """Return the set of image_ids matching the given query."""
        idx = _get_index(stage)
        result: set[str] = set()

        if stage == "detect":
            cls_universe = idx["classes"] if not classes else classes
            sm_list = source_models or ["__any__"]
            for cn in cls_universe:
                if source_models:
                    for sm in source_models:
                        result |= idx["by_class_model"].get((cn, sm), set())
                else:
                    result |= idx["by_class"].get(cn, set())
            return result

        if stage == "filter":
            cls_universe = idx["classes"] if not classes else classes
            sts = statuses or ["auto_accepted", "needs_evaluation", "dropped"]
            for st in sts:
                if st == "auto_accepted":
                    for cn in cls_universe:
                        result |= idx["auto_accepted"].get(cn, set())
                elif st == "needs_evaluation":
                    for cn in cls_universe:
                        result |= idx["needs_evaluation"].get(cn, set())
                elif st == "dropped":
                    rsn_iter = reasons or _FILTER_DROP_REASONS
                    for cn in cls_universe:
                        for rsn in rsn_iter:
                            result |= idx["dropped"].get((rsn, cn), set())
            return result

        if stage == "evaluate":
            # Pair-query short-circuit
            if relabel_from and relabel_to:
                return set(idx["relabel_pairs"].get((relabel_from, relabel_to), set()))
            sts = statuses or ["accepted", "review", "rejected", "drops", "relabeled"]
            roles = [class_role] if class_role in ("original", "verdict") else ["original", "verdict"]
            for role in roles:
                book = idx["by_status_original"] if role == "original" else idx["by_status_verdict"]
                cls_universe = (
                    idx["classes_original"] if role == "original" else idx["classes_verdict"]
                )
                cls_iter = classes or cls_universe
                for st in sts:
                    for cn in cls_iter:
                        result |= book.get((st, cn), set())
            return result

        if stage == "finalize":
            cls_universe = idx["classes"] if not classes else classes
            sts = statuses or ["accepted", "review", "dropped"]
            for st in sts:
                book = idx.get(st) or {}
                for cn in cls_universe:
                    result |= book.get(cn, set())
            return result

        # refine — empty scaffold
        return set()

    @app.get("/api/search/schema")
    async def search_schema() -> dict[str, Any]:
        """Return the option universes per stage for the sidebar UI.

        Class lists come from the cached index when warm; otherwise fall
        back to ``classes.txt``. Statuses/reasons/source_models are
        static defaults plus anything seen in the index.
        """
        all_classes = _load_classes()

        def _classes_for(stage: str) -> list[str]:
            if stage in _class_index_cache:
                idx = _class_index_cache[stage]
                if stage == "evaluate":
                    return sorted(set(idx.get("classes_original", [])) | set(idx.get("classes_verdict", [])))
                return idx.get("classes") or all_classes
            return all_classes

        return {
            "stages": [
                {"id": "detect",   "label": "1. Proposals (detect)"},
                {"id": "filter",   "label": "2. Filter"},
                {"id": "evaluate", "label": "3. Evaluate"},
                {"id": "refine",   "label": "4. Refine"},
                {"id": "finalize", "label": "5. Finalize"},
            ],
            "detect": {
                "classes": _classes_for("detect"),
                "statuses": [],  # detect has no status; source_model is the dimension
                "source_models": list(_DETECT_SOURCE_MODELS),
            },
            "filter": {
                "classes": _classes_for("filter"),
                "statuses": ["auto_accepted", "needs_evaluation", "dropped"],
                "reasons": list(_FILTER_DROP_REASONS),
            },
            "evaluate": {
                "classes": _classes_for("evaluate"),
                "statuses": ["accepted", "review", "rejected", "drops", "relabeled"],
                "class_roles": ["original", "verdict", "either"],
            },
            "refine": {
                "classes": _classes_for("refine"),
                "statuses": [],
            },
            "finalize": {
                "classes": _classes_for("finalize"),
                "statuses": ["accepted", "review", "dropped"],
            },
        }

    @app.get("/api/search")
    async def search(
        stage: str = Query(..., pattern=r"^(detect|filter|evaluate|refine|finalize)$"),
        classes: str = Query("", description="Comma-separated class names; empty = any"),
        statuses: str = Query("", description="Comma-separated stage-specific statuses"),
        reasons: str = Query("", description="Filter drop reasons (filter+dropped only)"),
        class_role: str = Query("either", pattern=r"^(original|verdict|either)$"),
        relabel_from: str = Query("", description="Evaluate pair-query: original class"),
        relabel_to: str = Query("", description="Evaluate pair-query: verdict class"),
        source_models: str = Query("", description="Detect source models filter"),
        offset: int = Query(0, ge=0),
        limit: int = Query(500, ge=1, le=5000),
    ) -> dict[str, Any]:
        """Stage-aware class/status search → paginated image list.

        Response shape mirrors ``/api/stems`` so the frontend can swap the
        URL without changing item-rendering code.
        """
        if not db_path.exists():
            return {
                "items": [], "total": 0, "offset": offset, "limit": limit,
                "filters": {"statuses": [], "stages": list(PIPELINE_STAGES)},
            }

        ids = _resolve_search(
            stage,
            _csv(classes), _csv(statuses), _csv(reasons),
            class_role, relabel_from.strip(), relabel_to.strip(),
            _csv(source_models),
        )
        sorted_ids = sorted(ids)
        total = len(sorted_ids)
        page_ids = sorted_ids[offset : offset + limit]

        items: list[dict[str, Any]] = []
        if page_ids:
            ph = ",".join("?" * len(page_ids))
            rows = _query(
                f"SELECT image_id, status, stages_completed FROM image_meta "
                f"WHERE image_id IN ({ph})",
                tuple(page_ids),
            )
            by_id = {r["image_id"]: r for r in rows}
            for img in page_ids:
                r = by_id.get(img) or {}
                raw = r.get("stages_completed") or "[]"
                try:
                    stages = json.loads(raw) if isinstance(raw, str) else raw
                    if not isinstance(stages, list):
                        stages = []
                except (json.JSONDecodeError, TypeError):
                    stages = []
                items.append({
                    "image_id": img,
                    "status": r.get("status") or "",
                    "stages_completed": stages,
                })
        return {
            "items": items, "total": total, "offset": offset, "limit": limit,
            "filters": {"statuses": [], "stages": list(PIPELINE_STAGES)},
        }

    @app.get("/api/stems")
    async def get_stems(
        offset: int = Query(0, ge=0),
        limit: int = Query(500, ge=1, le=5000),
        q: str = Query("", description="Substring match on image_id"),
        status: str = Query("", description="Exact image_meta.status filter"),
        stage: str = Query(
            "",
            description="Only images whose stages_completed contains this stage",
        ),
    ) -> dict[str, Any]:
        """Return a paginated slice of image_ids with status and stages_completed.

        Response shape::

            {
              "items":  [{image_id, status, stages_completed: [...]}, ...],
              "total":  matching-row count,
              "offset": echo of offset,
              "limit":  echo of limit,
              "filters": {
                "statuses": [...distinct image_meta.status values...],
                "stages":   [...canonical pipeline stage names...]
              }
            }
        """
        empty: dict[str, Any] = {
            "items": [],
            "total": 0,
            "offset": offset,
            "limit": limit,
            "filters": {"statuses": [], "stages": list(PIPELINE_STAGES)},
        }
        if not db_path.exists():
            return empty

        where: list[str] = []
        params: list[Any] = []
        if q:
            where.append("image_id LIKE ?")
            params.append(f"%{q}%")
        if status:
            where.append("status = ?")
            params.append(status)
        if stage:
            # stages_completed is a JSON array stored as TEXT; match the literal
            # "stage_name" token so "detect" doesn't match "detect_merge".
            where.append("stages_completed LIKE ?")
            params.append(f'%"{stage}"%')
        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        total_row = _query(
            f"SELECT COUNT(*) AS n FROM image_meta {where_sql}",
            tuple(params),
        )
        total = int(total_row[0]["n"]) if total_row else 0

        rows = _query(
            f"""
            SELECT image_id, status, stages_completed
            FROM image_meta
            {where_sql}
            ORDER BY image_id
            LIMIT ? OFFSET ?
            """,
            tuple(params + [limit, offset]),
        )

        items: list[dict[str, Any]] = []
        for r in rows:
            raw = r.get("stages_completed") or "[]"
            try:
                stages = json.loads(raw) if isinstance(raw, str) else raw
                if not isinstance(stages, list):
                    stages = []
            except (json.JSONDecodeError, TypeError):
                stages = []
            items.append(
                {
                    "image_id": r["image_id"],
                    "status": r.get("status") or "",
                    "stages_completed": stages,
                }
            )

        status_rows = _query(
            "SELECT DISTINCT status FROM image_meta WHERE status IS NOT NULL AND status <> '' ORDER BY status"
        )
        statuses = [s["status"] for s in status_rows]

        return {
            "items": items,
            "total": total,
            "offset": offset,
            "limit": limit,
            "filters": {"statuses": statuses, "stages": list(PIPELINE_STAGES)},
        }

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

        # Stage checkpoints (detect, filter, evaluate, refine, finalize).
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
            "filter": stages_data.get("filter"),
            "evaluate": stages_data.get("evaluate"),
            "refine": stages_data.get("refine"),
            "finalize": stages_data.get("finalize"),
            "human_review": stages_data.get("human_review"),
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
