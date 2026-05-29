"""
Embed DataTang with DINOv3-giant, FAISS-dedup, then pick the most diverse N.

Pipeline:
    1. Compute DINOv3-giant CLS embeddings for every image, cached as
       per-image .npy via the existing embedding_cache utility.
    2. FAISS near-duplicate dedup (greedy, threshold-based) — survivors
       are the candidate pool.
    3. Farthest-Point Sampling (FPS) on the survivors' embeddings to
       select exactly N images that maximally cover the embedding
       manifold. This is the "aggression max" step — among all subsets
       of size N, FPS approximates the one with the largest minimum
       pairwise distance.
    4. Write a manifest (selected/dedup_drops/fps_drops) for downstream
       viewer + YOLO subset builders.

Manifest schema (JSON):
    {
        "input_dir":    str,
        "embedding":    {"model_id": str, "stage": str, "dim": int},
        "dedup":        {"threshold": float, "k_neighbors": int,
                         "total": int, "survivors": int, "drops": int},
        "selection":    {"target": int, "selected": int,
                         "method": "farthest_point_sampling"},
        "selected":     [stem, ...],          # final keep set
        "dedup_drops":  [stem, ...],          # removed by FAISS dedup
        "fps_drops":    [stem, ...],          # survived dedup but not picked by FPS
    }

Usage:
    python -m scripts.dataset_selection.select_diverse_subset \
        --input-dir /media/data_2/datasets/datasets_pavan/DataTang_val \
        --out-dir output/dataset_selection/datatang_diverse_1000 \
        --target 1000 \
        --dedup-threshold 0.9 \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from data_miner.config import DINO_MODELS, DeduplicationConfig, DinoEmbeddingStage
from data_miner.modules.deduplicator import Deduplicator


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect_images(input_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(input_dir.rglob(f"*{ext}"))
    return sorted(paths)


def farthest_point_sampling(
    embeddings: np.ndarray,
    n: int,
    seed_idx: int = 0,
) -> list[int]:
    """Greedy FPS on L2-normalized embeddings.

    Maintains, for every point, its distance to the nearest already-selected
    point. At each step pick the point whose nearest-selected distance is
    largest. Cosine distance ≡ 1 - inner-product on normalized vectors.

    O(n * N) time, O(N) memory. For N≈11k and n=1000 this is ~11M ops,
    < 1 sec on CPU.
    """
    embs = embeddings.astype(np.float32, copy=False)
    faiss.normalize_L2(embs)

    N = embs.shape[0]
    if n >= N:
        return list(range(N))

    selected = [seed_idx]
    # Distance from every point to its nearest selected point (cosine distance)
    sims_to_selected = embs @ embs[seed_idx]
    min_dist = 1.0 - sims_to_selected  # shape (N,)

    for _ in range(1, n):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        new_sims = embs @ embs[next_idx]
        new_dist = 1.0 - new_sims
        np.minimum(min_dist, new_dist, out=min_dist)
        # Mark the selected point as already-picked so we never re-pick it
        min_dist[next_idx] = -np.inf

    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, required=True,
                    help="Directory of images (recursively scanned)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Where to write manifest.json + dedup output dir")
    ap.add_argument("--target", type=int, default=1000,
                    help="Number of images to keep after FPS")
    ap.add_argument("--dino-model", default="dinov3-giant",
                    choices=list(DINO_MODELS.keys()),
                    help="Which DINOv3 variant to use for embeddings")
    ap.add_argument("--dedup-threshold", type=float, default=0.9,
                    help="Cosine-similarity threshold for FAISS dedup "
                         "(higher = stricter, fewer drops)")
    ap.add_argument("--k-neighbors", type=int, default=1000,
                    help="FAISS k-NN search width during dedup")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="DINO inference batch size (giant=7B, keep small)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ignore-cache", action="store_true",
                    help="Force recomputation of embeddings")
    args = ap.parse_args()

    input_dir = args.input_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_dir)
    if not images:
        raise SystemExit(f"No images found under {input_dir}")
    print(f"[1/3] Found {len(images)} images under {input_dir}")

    dedup_conf = DeduplicationConfig(
        output_dir=out_dir / "dedup_survivors",
        dino_model_id=DINO_MODELS[args.dino_model],
        device=args.device,
        batch_size=args.batch_size,
        threshold=args.dedup_threshold,
        k_neighbors=args.k_neighbors,
        dino_embedding_stage=DinoEmbeddingStage.HIDDEN_CLS,
        cache_embeddings=True,
        ignore_cache=args.ignore_cache,
    )

    dedup = Deduplicator(dedup_conf, device_map=dedup_conf.device)
    print(f"[2/3] DINOv3 embed + FAISS dedup at threshold={args.dedup_threshold}")
    result = dedup.deduplicate(
        frame_paths=images,
        copy_files=False,
        input_dir=input_dir,
    )
    survivor_paths = result.unique_paths
    survivor_stems = [p.stem for p in survivor_paths]
    survivor_set = set(survivor_stems)
    dedup_drop_stems = [p.stem for p in images if p.stem not in survivor_set]
    print(f"      survivors={result.unique_frames}, "
          f"drops={result.duplicates_removed} "
          f"({result.dedup_rate:.1%})")

    # Free the DINO model before FPS (FPS is CPU-only and we don't want to
    # hold giant weights in VRAM longer than necessary).
    dedup.unload_model()

    # Re-load survivor embeddings from cache (per-stem .npy)
    cache_dir = dedup._get_cache_dir(input_dir)
    if cache_dir is None:
        raise SystemExit("Embedding cache disabled — cannot run FPS")
    survivor_embs = np.stack([
        np.load(cache_dir / f"{stem}.npy") for stem in survivor_stems
    ]).astype(np.float32)

    print(f"[3/3] FPS selection of {args.target} from {len(survivor_stems)} survivors")
    if args.target >= len(survivor_stems):
        selected_stems = list(survivor_stems)
        fps_drop_stems: list[str] = []
        print(f"      target >= survivors, keeping all {len(survivor_stems)}")
    else:
        selected_idx = farthest_point_sampling(survivor_embs, args.target)
        selected_set = {survivor_stems[i] for i in selected_idx}
        selected_stems = [survivor_stems[i] for i in selected_idx]
        fps_drop_stems = [s for s in survivor_stems if s not in selected_set]

    manifest = {
        "input_dir": str(input_dir),
        "embedding": {
            "model_id": dedup_conf.dino_model_id,
            "stage": dedup_conf.dino_embedding_stage.value,
            "dim": int(survivor_embs.shape[1]),
            "cache_dir": str(cache_dir),
        },
        "dedup": {
            "threshold": args.dedup_threshold,
            "k_neighbors": args.k_neighbors,
            "total": len(images),
            "survivors": len(survivor_stems),
            "drops": len(dedup_drop_stems),
        },
        "selection": {
            "target": args.target,
            "selected": len(selected_stems),
            "method": "farthest_point_sampling",
        },
        "selected": selected_stems,
        "dedup_drops": dedup_drop_stems,
        "fps_drops": fps_drop_stems,
    }

    # Also persist the path-stem map so downstream tools don't have to
    # re-scan the input dir.
    stem_to_path = {p.stem: str(p) for p in images}
    (out_dir / "stem_to_path.json").write_text(json.dumps(stem_to_path, indent=2))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote manifest: {out_dir / 'manifest.json'}")
    print(f"  total      : {len(images)}")
    print(f"  dedup drop : {len(dedup_drop_stems)}")
    print(f"  fps drop   : {len(fps_drop_stems)}")
    print(f"  selected   : {len(selected_stems)}")


if __name__ == "__main__":
    main()
