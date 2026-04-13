"""FastAPI backend for the pipeline viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


def _load_json(path: Path) -> Any:
    if path.exists():
        return json.loads(path.read_text())
    return None


def _discover_stems(output_dir: Path) -> list[str]:
    ckpt_dir = output_dir / ".checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted(d.name for d in ckpt_dir.iterdir() if d.is_dir())


def _find_image_path(
    stem: str, output_dir: Path, image_dir: Path | None
) -> Path | None:
    trace_path = output_dir / "traces" / f"{stem}.json"
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
        img_path = Path(trace.get("image_path", ""))
        if img_path.exists():
            return img_path
    if image_dir and image_dir.exists():
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = image_dir / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def create_app(output_dir: Path, image_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="AA v2 Viewer")

    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = static_dir / "index.html"
        return html_path.read_text()

    @app.get("/api/stems")
    async def get_stems():
        return _discover_stems(output_dir)

    @app.get("/api/data/{stem}")
    async def get_data(stem: str):
        ckpt = output_dir / ".checkpoints" / stem
        if not ckpt.exists():
            raise HTTPException(404, f"No checkpoint for {stem}")

        trace_path = output_dir / "traces" / f"{stem}.json"
        label_path = output_dir / "labels" / f"{stem}.txt"

        data: dict[str, Any] = {
            "proposal": _load_json(ckpt / "proposal.json"),
            "filtering": _load_json(ckpt / "filtering.json"),
            "vlm_reasoning": _load_json(ckpt / "vlm_reasoning.json"),
            "vlm_refinement": _load_json(ckpt / "vlm_refinement.json"),
            "vlm_validation": _load_json(ckpt / "vlm_validation.json"),
            "trace": _load_json(trace_path),
            "labels": label_path.read_text().strip() if label_path.exists() else "",
        }

        # Find image and get dimensions
        img_path = _find_image_path(stem, output_dir, image_dir)
        if img_path:
            from PIL import Image

            with Image.open(img_path) as img:
                data["image_width"], data["image_height"] = img.size
            data["image_url"] = f"/api/image/{stem}"
        else:
            data["image_width"] = 0
            data["image_height"] = 0
            data["image_url"] = None

        return data

    @app.get("/api/image/{stem}")
    async def get_image(stem: str):
        img_path = _find_image_path(stem, output_dir, image_dir)
        if not img_path or not img_path.exists():
            raise HTTPException(404, f"Image not found for {stem}")
        media_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(img_path.suffix.lower(), "image/jpeg")
        return FileResponse(img_path, media_type=media_type)

    return app
