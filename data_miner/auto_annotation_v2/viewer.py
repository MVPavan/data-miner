"""Interactive NiceGUI viewer for auto_annotation_v2 pipeline results.

Usage:
    python -m data_miner.auto_annotation_v2.viewer --output-dir output/auto_annotation_v2
    python -m data_miner.auto_annotation_v2.viewer --image output/sample/fl_pj_sample/img.jpg
    python -m data_miner.auto_annotation_v2.viewer --image-dir output/sample/fl_pj_sample
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from nicegui import app, ui

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

SOURCE_COLORS = {
    "falcon": "#0096FF",
    "grounding_dino": "#FF6400",
    "sam": "#00C864",
}
DECISION_COLORS = {
    "accept": "#00C800",
    "needs_review": "#FFA500",
    "reject": "#FF0000",
}
ACTION_COLORS = {
    "accept": "#00C800",
    "human_review": "#FFA500",
    "reject": "#FF0000",
}
FALLBACK_COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#800000", "#008000", "#000080", "#808000", "#800080", "#008080",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    if path.exists():
        return json.loads(path.read_text())
    return None


def _discover_images(output_dir: Path) -> list[str]:
    """Find all image stems that have checkpoint data."""
    ckpt_dir = output_dir / ".checkpoints"
    if not ckpt_dir.exists():
        return []
    stems = sorted(d.name for d in ckpt_dir.iterdir() if d.is_dir())
    return stems


def _find_image_path(stem: str, output_dir: Path, image_dir: Path | None) -> Path | None:
    """Try to find the original image file given its stem."""
    # Check trace for original path
    trace_path = output_dir / "traces" / f"{stem}.json"
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())
        img_path = Path(trace.get("image_path", ""))
        if img_path.exists():
            return img_path

    # Search image_dir
    if image_dir and image_dir.exists():
        for ext in (".jpg", ".jpeg", ".png"):
            p = image_dir / f"{stem}{ext}"
            if p.exists():
                return p

    return None


def _load_stage_data(stem: str, output_dir: Path) -> dict[str, Any]:
    """Load all checkpoint data for an image."""
    ckpt = output_dir / ".checkpoints" / stem
    trace_path = output_dir / "traces" / f"{stem}.json"
    label_path = output_dir / "labels" / f"{stem}.txt"

    data: dict[str, Any] = {}
    data["proposal"] = _load_json(ckpt / "proposal.json")
    data["filtering"] = _load_json(ckpt / "filtering.json")
    data["vlm_reasoning"] = _load_json(ckpt / "vlm_reasoning.json")
    data["vlm_refinement"] = _load_json(ckpt / "vlm_refinement.json")
    data["vlm_validation"] = _load_json(ckpt / "vlm_validation.json")
    data["trace"] = _load_json(trace_path)
    data["labels"] = label_path.read_text().strip() if label_path.exists() else ""
    return data


# ---------------------------------------------------------------------------
# SVG builders — each returns an SVG string for interactive_image.content
# ---------------------------------------------------------------------------


def _bbox_to_px(bbox: dict, w: int, h: int) -> tuple[float, float, float, float]:
    x1 = bbox["x1"] * w
    y1 = bbox["y1"] * h
    x2 = bbox["x2"] * w
    y2 = bbox["y2"] * h
    return x1, y1, x2, y2


def _svg_rect(x1: float, y1: float, x2: float, y2: float,
              color: str, label: str, idx: int) -> str:
    bw = x2 - x1
    bh = y2 - y1
    label_y = max(12, y1 - 4)
    return (
        f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
        f'fill="none" stroke="{color}" stroke-width="3" pointer-events="all" cursor="pointer" '
        f'data-idx="{idx}"/>'
        f'<text x="{x1:.1f}" y="{label_y:.1f}" fill="{color}" '
        f'font-size="13" font-weight="bold" '
        f'style="text-shadow: 1px 1px 2px black, -1px -1px 2px black;">'
        f'{label}</text>'
    )


def build_proposal_svg(candidates: list[dict], w: int, h: int,
                       cid_index: dict[str, int] | None = None) -> str:
    parts: list[str] = []
    for i, c in enumerate(candidates):
        x1, y1, x2, y2 = _bbox_to_px(c["bbox"], w, h)
        source = c.get("source_model", "unknown")
        color = SOURCE_COLORS.get(source, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        score = c.get("score", 0)
        idx = cid_index.get(c.get("candidate_id", ""), i) if cid_index else i
        label = f"#{idx} {source} {score:.2f}"
        parts.append(_svg_rect(x1, y1, x2, y2, color, label, idx))
    return "\n".join(parts)


def build_filtering_svg(candidates: list[dict], w: int, h: int,
                        cid_index: dict[str, int] | None = None) -> str:
    parts: list[str] = []
    for i, c in enumerate(candidates):
        x1, y1, x2, y2 = _bbox_to_px(c["bbox"], w, h)
        source = c.get("source_model", "unknown")
        color = SOURCE_COLORS.get(source, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])
        idx = cid_index.get(c.get("candidate_id", ""), i) if cid_index else i
        label = f"#{idx} {source}"
        parts.append(_svg_rect(x1, y1, x2, y2, color, label, idx))
    return "\n".join(parts)


def build_screening_svg(candidates: list[dict], verdicts: list[dict],
                        w: int, h: int,
                        cid_index: dict[str, int] | None = None) -> str:
    verdict_map = {v["candidate_id"]: v for v in verdicts}
    cand_map = {c["candidate_id"]: c for c in candidates}
    parts: list[str] = []
    for i, (cid, v) in enumerate(verdict_map.items()):
        c = cand_map.get(cid)
        if not c:
            continue
        x1, y1, x2, y2 = _bbox_to_px(c["bbox"], w, h)
        decision = v.get("decision", "reject")
        conf = v.get("confidence", 0)
        color = DECISION_COLORS.get(decision, "#888888")
        idx = cid_index.get(cid, i) if cid_index else i
        label = f"#{idx} {decision.upper()} {conf:.2f}"
        parts.append(_svg_rect(x1, y1, x2, y2, color, label, idx))
    return "\n".join(parts)


def build_detailed_svg(candidates: list[dict], verdicts: list[dict],
                       w: int, h: int,
                       cid_index: dict[str, int] | None = None) -> str:
    return build_screening_svg(candidates, verdicts, w, h, cid_index)


def build_refinement_svg(candidates: list[dict], w: int, h: int,
                         cid_index: dict[str, int] | None = None) -> str:
    return build_filtering_svg(candidates, w, h, cid_index)


def build_final_svg(annotations: list[dict], w: int, h: int,
                    cid_index: dict[str, int] | None = None) -> str:
    parts: list[str] = []
    for i, a in enumerate(annotations):
        x1, y1, x2, y2 = _bbox_to_px(a["bbox"], w, h)
        action = a.get("action", "reject")
        conf = a.get("confidence", 0)
        color = ACTION_COLORS.get(action, "#888888")
        idx = cid_index.get(a.get("candidate_id", ""), i) if cid_index else i
        label = f"#{idx} {action.upper()} {conf:.2f}"
        parts.append(_svg_rect(x1, y1, x2, y2, color, label, idx))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Detail panel builders
# ---------------------------------------------------------------------------


def _format_candidate_details(c: dict, cid_index: dict[str, int] | None = None) -> str:
    bbox = c.get("bbox", {})
    cid = c.get("candidate_id", "?")
    idx = cid_index.get(cid, "?") if cid_index else "?"
    return (
        f"**#{idx}** | **ID:** {cid}\n\n"
        f"**Class:** {c.get('class_name', '?')} | **Label:** {c.get('label', '?')}\n\n"
        f"**Model:** {c.get('source_model', '?')} | **Score:** {c.get('score', 0):.4f}\n\n"
        f"**BBox:** ({bbox.get('x1', 0):.4f}, {bbox.get('y1', 0):.4f}, "
        f"{bbox.get('x2', 0):.4f}, {bbox.get('y2', 0):.4f})\n\n"
        f"**Expression:** {c.get('expression', '?')}"
    )


def _format_verdict_details(v: dict, cid_index: dict[str, int] | None = None) -> str:
    cid = v.get("candidate_id", "?")
    idx = cid_index.get(cid, "?") if cid_index else "?"
    return (
        f"**#{idx}** | **ID:** {cid}\n\n"
        f"**Decision:** {v.get('decision', '?')}\n\n"
        f"**Confidence:** {v.get('confidence', 0):.4f}\n\n"
        f"**Reasoning:** {v.get('reasoning', 'N/A')}"
    )


def _format_annotation_details(a: dict, cid_index: dict[str, int] | None = None) -> str:
    trace = a.get("reasoning_trace", [])
    cid = a.get("candidate_id", "?")
    idx = cid_index.get(cid, "?") if cid_index else "?"
    return (
        f"**#{idx}** | **ID:** {cid}\n\n"
        f"**Action:** {a.get('action', '?')}\n\n"
        f"**Confidence:** {a.get('confidence', 0):.4f}\n\n"
        f"**Source:** {a.get('source_model', '?')}\n\n"
        f"**Refined:** {a.get('was_refined', False)}\n\n"
        f"**Trace:**\n" + "\n".join(f"- {t}" for t in trace)
    )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------


class ViewerState:
    """Shared state for the viewer."""

    def __init__(self, output_dir: Path, image_dir: Path | None):
        self.output_dir = output_dir
        self.image_dir = image_dir
        self.stems = _discover_images(output_dir)
        self.current_stem: str | None = self.stems[0] if self.stems else None
        self.current_stage: str = "proposal"
        self.stage_data: dict[str, Any] = {}
        self.image_path: Path | None = None
        self.image_size: tuple[int, int] = (0, 0)
        # Stable mapping: candidate_id → proposal-order index (persists across stages)
        self.cid_index: dict[str, int] = {}

    def load_image(self, stem: str) -> None:
        self.current_stem = stem
        self.stage_data = _load_stage_data(stem, self.output_dir)
        self.image_path = _find_image_path(stem, self.output_dir, self.image_dir)
        if self.image_path and self.image_path.exists():
            from PIL import Image
            img = Image.open(self.image_path)
            self.image_size = img.size
            img.close()
        # Build stable candidate_id → index map from proposal order
        self.cid_index = {}
        proposals = self.stage_data.get("proposal") or []
        for i, c in enumerate(proposals):
            cid = c.get("candidate_id")
            if cid:
                self.cid_index[cid] = i


def _get_stage_tabs() -> list[tuple[str, str]]:
    return [
        ("proposal", "1. Proposal"),
        ("filtering", "2. Filtering"),
        ("screening", "3a. Screening"),
        ("detailed", "3b. Detailed"),
        ("refinement", "4. Refinement"),
        ("validation", "5. Validation"),
        ("final", "6. Final"),
    ]


def build_ui(state: ViewerState) -> None:
    """Build the NiceGUI page."""

    # Serve image files
    if state.image_dir:
        app.add_static_files("/images", str(state.image_dir))

    # Track UI elements we need to update
    image_ref: dict[str, Any] = {}
    detail_ref: dict[str, Any] = {}

    def get_svg() -> str:
        """Build SVG for the current stage."""
        d = state.stage_data
        w, h = state.image_size
        cidx = state.cid_index
        if w == 0 or h == 0:
            return ""

        stage = state.current_stage

        if stage == "proposal" and d.get("proposal"):
            return build_proposal_svg(d["proposal"], w, h, cidx)

        if stage == "filtering" and d.get("filtering"):
            return build_filtering_svg(d["filtering"], w, h, cidx)

        if stage == "screening":
            reasoning = d.get("vlm_reasoning") or {}
            filtered = d.get("filtering") or []
            screening = reasoning.get("screening", [])
            if screening:
                return build_screening_svg(filtered, screening, w, h, cidx)

        if stage == "detailed":
            reasoning = d.get("vlm_reasoning") or {}
            filtered = d.get("filtering") or []
            detailed = reasoning.get("detailed", [])
            if detailed:
                return build_detailed_svg(filtered, detailed, w, h, cidx)

        if stage == "refinement":
            ref_data = d.get("vlm_refinement") or {}
            candidates = ref_data.get("candidates", [])
            if candidates:
                return build_refinement_svg(candidates, w, h, cidx)

        if stage == "validation":
            val_data = d.get("vlm_validation") or {}
            ref_data = d.get("vlm_refinement") or {}
            filtered = d.get("filtering") or []
            val_screening = val_data.get("screening", [])
            cands = ref_data.get("candidates", filtered)
            if val_screening:
                return build_screening_svg(cands, val_screening, w, h, cidx)

        if stage == "final":
            trace = d.get("trace") or {}
            annotations = trace.get("final_annotations", [])
            if annotations:
                return build_final_svg(annotations, w, h, cidx)

        return ""

    def get_detail_text() -> str:
        """Build detail panel content for current stage."""
        d = state.stage_data
        stage = state.current_stage
        cidx = state.cid_index

        if stage == "proposal" and d.get("proposal"):
            cands = d["proposal"]
            return f"**{len(cands)} candidates** from detection models\n\n" + "\n\n---\n\n".join(
                _format_candidate_details(c, cidx) for c in cands
            )

        if stage == "filtering" and d.get("filtering"):
            cands = d["filtering"]
            proposal_count = len(d.get("proposal") or [])
            return f"**{proposal_count} → {len(cands)}** after geometric + IoU filtering\n\n" + "\n\n---\n\n".join(
                _format_candidate_details(c, cidx) for c in cands
            )

        if stage == "screening":
            reasoning = d.get("vlm_reasoning") or {}
            verdicts = reasoning.get("screening", [])
            if verdicts:
                n_accept = sum(1 for v in verdicts if v["decision"] == "accept")
                n_reject = sum(1 for v in verdicts if v["decision"] == "reject")
                n_review = sum(1 for v in verdicts if v["decision"] == "needs_review")
                header = f"**Screening:** {n_accept} accept, {n_review} review, {n_reject} reject\n\n"
                return header + "\n\n---\n\n".join(_format_verdict_details(v, cidx) for v in verdicts)

        if stage == "detailed":
            reasoning = d.get("vlm_reasoning") or {}
            verdicts = reasoning.get("detailed", [])
            if verdicts:
                return f"**{len(verdicts)} detailed verdicts**\n\n" + "\n\n---\n\n".join(
                    _format_verdict_details(v, cidx) for v in verdicts
                )
            return "No candidates needed detailed review."

        if stage == "refinement":
            ref_data = d.get("vlm_refinement") or {}
            candidates = ref_data.get("candidates", [])
            actions = ref_data.get("actions", [])
            if candidates:
                return f"**{len(candidates)} candidates, {len(actions)} actions**\n\n" + "\n\n---\n\n".join(
                    _format_candidate_details(c, cidx) for c in candidates
                )
            return "No candidates needed refinement."

        if stage == "validation":
            val_data = d.get("vlm_validation") or {}
            verdicts = val_data.get("screening", [])
            if verdicts:
                return f"**{len(verdicts)} validation verdicts**\n\n" + "\n\n---\n\n".join(
                    _format_verdict_details(v, cidx) for v in verdicts
                )
            return "No refined candidates to validate."

        if stage == "final":
            trace = d.get("trace") or {}
            annotations = trace.get("final_annotations", [])
            labels = d.get("labels", "")
            n_labels = len(labels.split("\n")) if labels.strip() else 0
            if annotations:
                return (
                    f"**{n_labels} YOLO labels saved**\n\n"
                    + "\n\n---\n\n".join(_format_annotation_details(a, cidx) for a in annotations)
                )
            return "No final annotations."

        return "No data for this stage."

    def refresh_image():
        """Update the image and SVG overlay."""
        if "img" in image_ref and state.image_path:
            image_ref["img"].set_source(str(state.image_path))
            image_ref["img"].set_content(get_svg())
        if "detail" in detail_ref:
            detail_ref["detail"].set_content(get_detail_text())

    def on_image_select(stem: str):
        state.load_image(stem)
        refresh_image()

    def on_stage_change(stage: str):
        state.current_stage = stage
        refresh_image()

    # --- Page layout ---
    ui.dark_mode(True)

    with ui.header().classes("bg-gray-900 text-white items-center"):
        ui.label("Auto Annotation v2 — Pipeline Viewer").classes("text-xl font-bold")
        ui.space()
        ui.label().bind_text_from(state, "current_stem", lambda s: f"Image: {s or 'none'}")

    with ui.left_drawer(value=True).classes("bg-gray-800 text-white").props("width=260"):
        ui.label("Images").classes("text-lg font-bold mb-2")
        if not state.stems:
            ui.label("No processed images found.").classes("text-gray-400")
        else:
            for stem in state.stems:
                ui.button(
                    stem[:30] + ("..." if len(stem) > 30 else ""),
                    on_click=lambda s=stem: on_image_select(s),
                ).classes("w-full text-left mb-1").props("flat dense")

    with ui.column().classes("w-full p-4"):
        # Stage tabs
        with ui.tabs().classes("w-full") as tabs:
            tab_map = {}
            for key, label in _get_stage_tabs():
                tab_map[key] = ui.tab(key, label=label)

        tabs.on_value_change(lambda e: on_stage_change(e.value))

        # Legend (fixed at top, below tabs)
        with ui.row().classes("w-full gap-4 items-center mt-2 mb-2 flex-wrap"):
            ui.label("Legend:").classes("font-bold")
            for name, color in SOURCE_COLORS.items():
                with ui.row().classes("items-center gap-1"):
                    ui.html(f'<div style="width:16px;height:16px;background:{color};border-radius:3px;"></div>')
                    ui.label(name).classes("text-sm")
            ui.label("|").classes("text-gray-500")
            for name, color in DECISION_COLORS.items():
                with ui.row().classes("items-center gap-1"):
                    ui.html(f'<div style="width:16px;height:16px;background:{color};border-radius:3px;"></div>')
                    ui.label(name).classes("text-sm")

        # Main content: image + detail panel side by side
        with ui.row().classes("w-full items-start gap-4 mt-2"):
            # Image panel
            with ui.column().classes("flex-grow"):
                if state.image_path and state.image_path.exists():
                    img_widget = ui.interactive_image(
                        str(state.image_path),
                        content=get_svg(),
                        sanitize=False,
                    ).classes("w-full max-w-4xl")
                    image_ref["img"] = img_widget
                else:
                    ui.label("No image loaded").classes("text-gray-400 text-xl")
                    img_widget = ui.interactive_image(
                        size=(800, 600),
                        sanitize=False,
                    ).classes("w-full max-w-4xl bg-gray-700")
                    image_ref["img"] = img_widget

            # Detail panel
            with ui.column().classes("w-96 min-w-80"):
                ui.label("Details").classes("text-lg font-bold mb-2")
                with ui.card().classes("w-full bg-gray-800"):
                    detail_md = ui.markdown(get_detail_text()).classes("text-sm")
                    detail_ref["detail"] = detail_md


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Interactive pipeline result viewer")
    parser.add_argument(
        "--output-dir", type=str, default="output/auto_annotation_v2",
        help="Pipeline output directory (default: output/auto_annotation_v2)",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Single image path (will use its directory for lookup)",
    )
    parser.add_argument(
        "--image-dir", type=str, default=None,
        help="Directory containing source images",
    )
    parser.add_argument(
        "--port", type=int, default=8090,
        help="Port to serve on (default: 8090)",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir) if args.image_dir else None

    # If single image specified, infer image_dir
    if args.image:
        img_path = Path(args.image)
        if image_dir is None:
            image_dir = img_path.parent

    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        sys.exit(1)

    state = ViewerState(output_dir, image_dir)

    if not state.stems:
        print(f"No checkpoint data found in {output_dir / '.checkpoints'}")
        sys.exit(1)

    # Load first image
    state.load_image(state.stems[0])

    @ui.page("/")
    def index():
        build_ui(state)

    print(f"Starting viewer at http://localhost:{args.port}")
    print(f"Output dir: {output_dir}")
    print(f"Image dir: {image_dir}")
    print(f"Found {len(state.stems)} processed images")

    ui.run(port=args.port, title="AA v2 Viewer", reload=False, show=False)


if __name__ == "__main__":
    main()
