"""FastAPI backend for the auto_annotation_v3 viewer.

Exposes per-image checkpoints (detect / evaluate / refine / meta), per-model
raw proposals, the YOLO label file, and the classes.txt manifest as JSON so
the single-page frontend can render every pipeline stage with bbox overlays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _load_json(path: Path) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _read_text(path: Path) -> str:
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""
    return ""


def _discover_image_ids(ckpt_dir: Path) -> list[str]:
    if not ckpt_dir.exists():
        return []
    return sorted(d.name for d in ckpt_dir.iterdir() if d.is_dir())


def _load_classes(job_dir: Path) -> list[str]:
    p = job_dir / "classes.txt"
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _load_class_id_map(job_dir: Path) -> dict[str, int]:
    """Build ``{class_name: class_id}`` from the frozen job config.

    ``pipeline.py`` writes ``{job_dir}/config.yaml`` as JSON. Falls back to
    positional indexing of ``classes.txt`` (which is wrong when ids are
    non-contiguous, but only used when config.yaml is missing).
    """
    cfg_path = job_dir / "config.yaml"
    if cfg_path.exists():
        data = _load_json(cfg_path)
        if isinstance(data, dict):
            reg = data.get("class_registry") or []
            if isinstance(reg, list):
                out: dict[str, int] = {}
                for c in reg:
                    if isinstance(c, dict) and "name" in c and "id" in c:
                        out[str(c["name"])] = int(c["id"])
                if out:
                    return out
    # Fallback — may produce wrong ids with gaps, logged by viewer UI.
    return {name: i for i, name in enumerate(_load_classes(job_dir))}


def _is_within(path: Path, roots: list[Path]) -> bool:
    """True if *path* resolves inside any of *roots* (after symlink resolution)."""
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


def _find_image_path(
    image_id: str, job_dir: Path, image_dir: Path | None, allowed_roots: list[Path]
) -> Path | None:
    """Locate the source image for *image_id*, constrained to *allowed_roots*.

    Resolution order:
      1. ``DetectResult.image_path`` from the per-image checkpoint, only if
         it resolves inside one of *allowed_roots*.
      2. ``{image_id}.{ext}`` inside *image_dir*.
    """
    detect = _load_json(job_dir / "checkpoints" / image_id / "detect.json")
    if isinstance(detect, dict):
        ip = detect.get("image_path")
        if ip:
            p = Path(ip)
            if p.exists() and _is_within(p, allowed_roots):
                return p

    if image_dir and image_dir.exists():
        for ext in IMAGE_EXTS:
            p = image_dir / f"{image_id}{ext}"
            if p.exists() and _is_within(p, allowed_roots):
                return p
    return None


def _list_proposals(ckpt_dir: Path, image_id: str) -> dict[str, Any]:
    """Return ``{model_name: ProposalResult dict}`` for all per-model files."""
    pdir = ckpt_dir / image_id / "proposals"
    out: dict[str, Any] = {}
    if not pdir.exists():
        return out
    for f in sorted(pdir.glob("*.json")):
        data = _load_json(f)
        if data is not None:
            out[f.stem] = data
    return out


def create_app(job_dir: Path, image_dir: Path | None = None) -> FastAPI:
    job_dir = Path(job_dir)
    ckpt_dir = job_dir / "checkpoints"

    # Image path allowlist — served files must resolve inside one of these.
    allowed_roots: list[Path] = [job_dir]
    if image_dir is not None:
        allowed_roots.append(image_dir)

    app = FastAPI(title="AA v3 Viewer")

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (static_dir / "index.html").read_text(encoding="utf-8")

    @app.get("/api/job")
    async def get_job_info() -> dict[str, Any]:
        return {
            "job_dir": str(job_dir),
            "classes": _load_classes(job_dir),
            "class_id_map": _load_class_id_map(job_dir),
            "summary": _load_json(job_dir / "summary.json"),
        }

    @app.get("/api/stems")
    async def get_stems() -> list[str]:
        return _discover_image_ids(ckpt_dir)

    @app.get("/api/data/{image_id}")
    async def get_data(image_id: str) -> dict[str, Any]:
        img_ckpt = ckpt_dir / image_id
        if not img_ckpt.exists():
            raise HTTPException(404, f"No checkpoint for {image_id}")

        data: dict[str, Any] = {
            "image_id": image_id,
            "meta": _load_json(img_ckpt / "meta.json"),
            "detect": _load_json(img_ckpt / "detect.json"),
            "evaluate": _load_json(img_ckpt / "evaluate.json"),
            "refine": _load_json(img_ckpt / "refine.json"),
            "proposals": _list_proposals(ckpt_dir, image_id),
            "trace": _load_json(job_dir / "traces" / f"{image_id}.json"),
            "review": _load_json(job_dir / "review" / f"{image_id}.json"),
            "labels": _read_text(job_dir / "labels" / f"{image_id}.txt"),
            "classes": _load_classes(job_dir),
            "class_id_map": _load_class_id_map(job_dir),
        }

        # Image dimensions: prefer detect's image_size, then probe file.
        img_w, img_h = 0, 0
        det = data["detect"]
        if isinstance(det, dict) and isinstance(det.get("image_size"), list) and len(det["image_size"]) == 2:
            img_w, img_h = int(det["image_size"][0]), int(det["image_size"][1])

        img_path = _find_image_path(image_id, job_dir, image_dir, allowed_roots)
        if img_path is not None and (img_w == 0 or img_h == 0):
            try:
                from PIL import Image

                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception:
                pass

        data["image_width"] = img_w
        data["image_height"] = img_h
        data["image_url"] = f"/api/image/{image_id}" if img_path else None
        return data

    @app.get("/api/image/{image_id}")
    async def get_image(image_id: str):
        img_path = _find_image_path(image_id, job_dir, image_dir, allowed_roots)
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
