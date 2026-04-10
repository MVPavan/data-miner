"""Visualize outputs of each pipeline stage for a given image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .contracts import (
    BoundingBox,
    VLMDecision,
)
from .utils import bbox_to_pixels

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

_DECISION_COLORS = {
    VLMDecision.ACCEPT: (0, 200, 0),
    VLMDecision.NEEDS_REVIEW: (255, 165, 0),
    VLMDecision.REJECT: (255, 0, 0),
}

_SOURCE_COLORS = {
    "falcon": (0, 150, 255),
    "grounding_dino": (255, 100, 0),
    "sam": (0, 200, 100),
}

_DEFAULT_COLOR = (255, 255, 0)


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except (OSError, IOError):
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Per-stage drawing
# ---------------------------------------------------------------------------


def _draw_candidates(
    image: Image.Image,
    candidates: list[dict],
    title: str,
    color_by: str = "source",
) -> Image.Image:
    """Draw candidates with labels. color_by: 'source' or 'index'."""
    rendered = image.copy().convert("RGB")
    draw = ImageDraw.Draw(rendered)
    font = _get_font(13)
    title_font = _get_font(18)

    # Title bar
    draw.rectangle([(0, 0), (rendered.width, 28)], fill=(0, 0, 0))
    draw.text((8, 5), title, fill=(255, 255, 255), font=title_font)

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (128, 128, 0),
        (128, 0, 128),
        (0, 128, 128),
    ]

    for idx, c in enumerate(candidates):
        bbox = c["bbox"]
        box = BoundingBox(x1=bbox["x1"], y1=bbox["y1"], x2=bbox["x2"], y2=bbox["y2"])
        px = bbox_to_pixels(box, rendered.size)

        if color_by == "source":
            color = _SOURCE_COLORS.get(
                c.get("source_model", ""), colors[idx % len(colors)]
            )
        else:
            color = colors[idx % len(colors)]

        draw.rectangle(px, outline=color, width=3)
        cid = c.get("candidate_id", f"[{idx}]")
        score = c.get("score")
        score_str = f" {score:.2f}" if isinstance(score, (int, float)) else ""
        label = f"{cid}{score_str}"
        draw.text((px[0], max(30, px[1] - 16)), label, fill=color, font=font)

    return rendered


def _draw_vlm_verdicts(
    image: Image.Image,
    candidates: list[dict],
    verdicts: list[dict],
    title: str,
) -> Image.Image:
    """Draw candidates colored by VLM decision (accept/reject/needs_review)."""
    rendered = image.copy().convert("RGB")
    draw = ImageDraw.Draw(rendered)
    font = _get_font(13)
    title_font = _get_font(18)

    draw.rectangle([(0, 0), (rendered.width, 28)], fill=(0, 0, 0))
    draw.text((8, 5), title, fill=(255, 255, 255), font=title_font)

    verdict_map = {v["candidate_id"]: v for v in verdicts}
    cand_map = {c["candidate_id"]: c for c in candidates}

    for cid, v in verdict_map.items():
        c = cand_map.get(cid)
        if c is None:
            continue
        bbox = c["bbox"]
        box = BoundingBox(x1=bbox["x1"], y1=bbox["y1"], x2=bbox["x2"], y2=bbox["y2"])
        px = bbox_to_pixels(box, rendered.size)

        decision = v.get("decision", "reject")
        color = _DECISION_COLORS.get(VLMDecision(decision), (128, 128, 128))
        conf = v.get("confidence", 0)
        draw.rectangle(px, outline=color, width=3)
        label = f"{decision} {conf:.2f}"
        draw.text((px[0], max(30, px[1] - 16)), label, fill=color, font=font)

    return rendered


def _draw_refinement(
    image: Image.Image,
    candidates: list[dict],
    title: str,
) -> Image.Image:
    """Draw refined candidates."""
    return _draw_candidates(image, candidates, title, color_by="source")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def visualize_stages(
    image_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """Generate per-stage visualizations. Returns list of saved images."""
    stem = image_path.stem
    ckpt_dir = checkpoint_dir / stem
    if not ckpt_dir.is_dir():
        print(f"No checkpoints found for {stem} at {ckpt_dir}")
        return []

    image = Image.open(image_path).convert("RGB")
    saved: list[Path] = []
    viz_dir = output_dir / "viz" / stem
    viz_dir.mkdir(parents=True, exist_ok=True)

    # 1. Proposal
    proposal_path = ckpt_dir / "proposal.json"
    if proposal_path.exists():
        candidates = json.loads(proposal_path.read_text())
        img = _draw_candidates(
            image,
            candidates,
            f"1. PROPOSAL — {len(candidates)} candidates",
            color_by="source",
        )
        out = viz_dir / "1_proposal.jpg"
        img.save(out, quality=90)
        saved.append(out)
        print(f"  [1] Proposal: {len(candidates)} candidates → {out}")

    # 2. Filtering
    filtering_path = ckpt_dir / "filtering.json"
    if filtering_path.exists():
        filtered = json.loads(filtering_path.read_text())
        img = _draw_candidates(
            image,
            filtered,
            f"2. FILTERING — {len(filtered)} survived",
            color_by="source",
        )
        out = viz_dir / "2_filtering.jpg"
        img.save(out, quality=90)
        saved.append(out)
        print(f"  [2] Filtering: {len(filtered)} survived → {out}")

    # 3. VLM Reasoning
    reasoning_path = ckpt_dir / "vlm_reasoning.json"
    filtered_cands = (
        json.loads(filtering_path.read_text()) if filtering_path.exists() else []
    )
    if reasoning_path.exists():
        data = json.loads(reasoning_path.read_text())
        screening = data.get("screening", [])
        detailed = data.get("detailed", [])

        if screening:
            img = _draw_vlm_verdicts(
                image,
                filtered_cands,
                screening,
                f"3a. SCREENING — {sum(1 for v in screening if v['decision'] == 'accept')} accept, "
                f"{sum(1 for v in screening if v['decision'] == 'needs_review')} review, "
                f"{sum(1 for v in screening if v['decision'] == 'reject')} reject",
            )
            out = viz_dir / "3a_screening.jpg"
            img.save(out, quality=90)
            saved.append(out)
            print(f"  [3a] Screening: {len(screening)} verdicts → {out}")

        if detailed:
            img = _draw_vlm_verdicts(
                image,
                filtered_cands,
                detailed,
                f"3b. DETAILED — {len(detailed)} reviewed",
            )
            out = viz_dir / "3b_detailed.jpg"
            img.save(out, quality=90)
            saved.append(out)
            print(f"  [3b] Detailed: {len(detailed)} verdicts → {out}")

    # 4. VLM Refinement
    refinement_path = ckpt_dir / "vlm_refinement.json"
    if refinement_path.exists():
        data = json.loads(refinement_path.read_text())
        refined = data.get("candidates", [])
        actions = data.get("actions", [])
        if refined:
            img = _draw_refinement(
                image,
                refined,
                f"4. REFINEMENT — {len(refined)} candidates, {len(actions)} actions",
            )
            out = viz_dir / "4_refinement.jpg"
            img.save(out, quality=90)
            saved.append(out)
            print(f"  [4] Refinement: {len(refined)} candidates → {out}")

    # 5. VLM Validation
    validation_path = ckpt_dir / "vlm_validation.json"
    if validation_path.exists():
        data = json.loads(validation_path.read_text())
        val_screening = data.get("screening", [])
        refined_cands = (
            json.loads(refinement_path.read_text()).get("candidates", [])
            if refinement_path.exists()
            else filtered_cands
        )
        if val_screening:
            img = _draw_vlm_verdicts(
                image,
                refined_cands,
                val_screening,
                f"5. VALIDATION — {len(val_screening)} verdicts",
            )
            out = viz_dir / "5_validation.jpg"
            img.save(out, quality=90)
            saved.append(out)
            print(f"  [5] Validation: {len(val_screening)} verdicts → {out}")

    # 6. Final (from trace)
    trace_dir = output_dir / "traces"
    trace_path = trace_dir / f"{stem}.json"
    label_path = output_dir / "labels" / f"{stem}.txt"
    if trace_path.exists():
        trace = json.loads(trace_path.read_text())

        # Draw accepted (green) + rejected (red) + review (orange) from final annotations
        rendered = image.copy().convert("RGB")
        draw_obj = ImageDraw.Draw(rendered)
        font = _get_font(13)
        title_font = _get_font(18)
        draw_obj.rectangle([(0, 0), (rendered.width, 28)], fill=(0, 0, 0))

        labels_text = label_path.read_text().strip() if label_path.exists() else ""
        n_labels = len(labels_text.split("\n")) if labels_text else 0

        # Build accept/reject/review sets from final_annotations
        final_anns = trace.get("final_annotations", [])
        accepted_ids = {
            a["candidate_id"] for a in final_anns if a["action"] == "accept"
        }
        rejected_ids = {
            a["candidate_id"] for a in final_anns if a["action"] == "reject"
        }
        review_ids = {
            a["candidate_id"] for a in final_anns if a["action"] == "human_review"
        }
        ann_map = {a["candidate_id"]: a for a in final_anns}

        draw_obj.text(
            (8, 5),
            f"6. FINAL — {len(accepted_ids)} accept, {len(rejected_ids)} reject, {len(review_ids)} review",
            fill=(255, 255, 255),
            font=title_font,
        )

        # Draw final annotations with their bboxes and status
        for ann in final_anns:
            cid = ann["candidate_id"]
            bbox = ann["bbox"]
            box = BoundingBox(
                x1=bbox["x1"], y1=bbox["y1"], x2=bbox["x2"], y2=bbox["y2"]
            )
            px = bbox_to_pixels(box, rendered.size)

            if cid in accepted_ids:
                color = (0, 200, 0)
                status = "ACCEPT"
            elif cid in rejected_ids:
                color = (255, 0, 0)
                status = "REJECT"
            else:
                color = (255, 165, 0)
                status = "REVIEW"

            draw_obj.rectangle(px, outline=color, width=3)
            conf = ann.get("confidence", 0)
            draw_obj.text(
                (px[0], max(30, px[1] - 16)),
                f"{status} {conf:.2f}",
                fill=color,
                font=font,
            )

        out = viz_dir / "6_final.jpg"
        rendered.save(out, quality=90)
        saved.append(out)
        print(
            f"  [6] Final: {len(accepted_ids)} accept, {len(rejected_ids)} reject, {len(review_ids)} review → {out}"
        )

    return saved


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Visualize auto_annotation_v2 stage outputs"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the input image"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/auto_annotation_v2",
        help="Pipeline output directory (default: output/auto_annotation_v2)",
    )
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / ".checkpoints"

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    print(f"Visualizing stages for: {image_path.name}")
    saved = visualize_stages(image_path, checkpoint_dir, output_dir)
    if saved:
        print(
            f"\n{len(saved)} visualizations saved to {output_dir / 'viz' / image_path.stem}/"
        )
    else:
        print("No visualizations produced. Run pipeline first.")


if __name__ == "__main__":
    main()
