"""Concatenate M1-M7 annotated outputs into per-image grids with mode labels."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path("/media/data_2/vlm/code/data_miner/output/dart_tests")
GRID_DIR = OUT / "concatenated_M1-M4"
GRID_DIR.mkdir(parents=True, exist_ok=True)

MODES = [
    "M1_baseline_seq",
    "M2_baseline_seq_detonly",
    "M3_fast_batched_fp16",
    "M4_fast_batched_detonly",
]

LABEL_H = 48
PAD = 6

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    TITLE_FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
except OSError:
    FONT = ImageFont.load_default()
    TITLE_FONT = FONT


def load_mode_image(mode: str, stem: str) -> Image.Image | None:
    p = OUT / mode / f"{stem}.jpg"
    if not p.exists():
        return None
    return Image.open(p).convert("RGB")


def make_tile(img: Image.Image, label: str) -> Image.Image:
    w, h = img.size
    tile = Image.new("RGB", (w, h + LABEL_H), "black")
    draw = ImageDraw.Draw(tile)
    bbox = draw.textbbox((0, 0), label, font=FONT)
    tx = (w - (bbox[2] - bbox[0])) // 2
    draw.text((tx, (LABEL_H - (bbox[3] - bbox[1])) // 2 - 2),
              label, fill="white", font=FONT)
    tile.paste(img, (0, LABEL_H))
    return tile


def image_stems() -> list[str]:
    first = OUT / MODES[0]
    return sorted(p.stem for p in first.glob("*.jpg"))


def build_grid(stem: str) -> Image.Image:
    tiles = []
    for m in MODES:
        img = load_mode_image(m, stem)
        if img is None:
            ref = next((t for t in tiles if t is not None), None)
            w = ref.size[0] if ref else 800
            h = (ref.size[1] - LABEL_H) if ref else 600
            ph = Image.new("RGB", (w, h + LABEL_H), "gray")
            d = ImageDraw.Draw(ph)
            d.text((20, 20), f"{m}\n(missing)", fill="white", font=FONT)
            tiles.append(ph)
        else:
            tiles.append(make_tile(img, m))

    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    tile_w = max(t.size[0] for t in tiles)
    tile_h = max(t.size[1] for t in tiles)
    title_h = 60
    grid_w = cols * tile_w + (cols + 1) * PAD
    grid_h = rows * tile_h + (rows + 1) * PAD + title_h
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((PAD, 10), f"Image: {stem}", fill="black", font=TITLE_FONT)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        x = PAD + c * (tile_w + PAD)
        y = title_h + PAD + r * (tile_h + PAD)
        grid.paste(t, (x, y))
    return grid


def main():
    stems = image_stems()
    print(f"Found {len(stems)} images, {len(MODES)} modes")
    for stem in stems:
        grid = build_grid(stem)
        out_path = GRID_DIR / f"{stem}_grid.jpg"
        grid.save(out_path, quality=92, subsampling=1)
        print(f"  saved {out_path} ({grid.size[0]}x{grid.size[1]})")
    print(f"\nAll grids in {GRID_DIR}")


if __name__ == "__main__":
    main()
