"""Threshold-sweep + per-video analytics for FAISS near-duplicate dedup.

Reads cached DINOv3 embeddings + a `source_map.json` (stem -> {source, video,
...}), runs greedy FAISS-style dedup at each threshold in the sweep, and
prints survivor counts per source and per video.

Greedy dedup semantics (matches data_miner.modules.deduplicator):
    Walk stems in sorted order. For each stem i, drop it iff some
    already-survivor j has cosine_sim(i, j) > threshold. Else keep.

Implementation: one (N, D) @ (D, N) matmul on L2-normalized embeddings.
For N=6875 / D=4096 / fp32 this is ~190 MB and ~1-2 s on CPU. Fast enough
to sweep many thresholds without re-embedding.

Usage:
    python -m scripts.dataset_selection.threshold_sweep \\
        --working-dir output/dataset_selection/combined_3ds \\
        --embeddings-dir /media/data_2/datasets/datasets_pavan/dinov3-vit7b16-pretrain-lvd1689m_hidden_cls_embeddings \\
        --thresholds 0.85 0.90 0.93 0.95 0.96 0.97 0.98 0.99 \\
        --top-videos 15
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_embeddings(stems: list[str], emb_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load .npy per stem; return (embeddings, surviving_stems_in_order).

    Stems missing from the cache are reported and skipped.
    """
    found: list[str] = []
    arrs: list[np.ndarray] = []
    missing = 0
    for s in stems:
        p = emb_dir / f"{s}.npy"
        if not p.is_file():
            missing += 1
            continue
        a = np.load(p)
        if a.ndim > 1:
            a = a.reshape(-1)
        arrs.append(a.astype(np.float32, copy=False))
        found.append(s)
    if missing:
        print(f"  WARN: {missing} stems missing from embedding cache", file=sys.stderr)
    if not arrs:
        raise SystemExit(f"no .npy embeddings found under {emb_dir}")
    embs = np.stack(arrs)
    # L2-normalize (in-place)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    embs /= norms
    return embs, found


def greedy_dedup_threshold_sweep(
    embs: np.ndarray, thresholds: list[float],
) -> dict[float, list[bool]]:
    """For each threshold t, compute a survivor mask of shape (N,).

    Greedy: walk i from 0 to N-1; survives iff no j<i with sim(i,j) > t survives.
    """
    N = embs.shape[0]
    # Single matmul for similarities. Shape (N, N), fp32.
    sims = embs @ embs.T

    out: dict[float, list[bool]] = {}
    for t in thresholds:
        survivors = np.ones(N, dtype=bool)
        # j < i lookup. Vectorize per row.
        for i in range(N):
            if not survivors[i]:
                continue
            # check earlier survivors with sim > t
            row = sims[i, :i]
            if row.size and (row > t).any():
                # Is the offender currently a survivor?
                # We must restrict to j where survivors[j]==True.
                if survivors[:i][row > t].any():
                    survivors[i] = False
        out[t] = survivors.tolist()
    return out


def analytics(
    survivor_mask: list[bool],
    stems: list[str],
    source_map: dict[str, dict],
    top_videos: int,
) -> dict:
    """Per-source + per-video survivor stats."""
    by_source = Counter()
    by_video = defaultdict(int)         # video name -> survivor count
    by_video_total = defaultdict(int)   # video name -> total count (denominator)

    for s, alive in zip(stems, survivor_mask):
        meta = source_map.get(s, {})
        src = meta.get("source", "?")
        vid = meta.get("video", "?")
        full_vid = f"{src}/{vid}"
        by_video_total[full_vid] += 1
        if alive:
            by_source[src] += 1
            by_video[full_vid] += 1

    # rank videos by drop ratio (1 - survived/total) descending
    ranked = sorted(
        by_video_total.keys(),
        key=lambda v: (by_video[v] / by_video_total[v], -by_video_total[v]),
    )

    return {
        "total_survivors": int(sum(survivor_mask)),
        "by_source": dict(by_source),
        "by_video_total": dict(by_video_total),
        "by_video_survivors": dict(by_video),
        "ranked_videos": ranked[:top_videos],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--working-dir", type=Path, required=True,
                    help="combined_3ds-style dir containing source_map.json")
    ap.add_argument("--embeddings-dir", type=Path, required=True,
                    help="Directory of <stem>.npy DINOv3 embeddings")
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.85, 0.90, 0.93, 0.95, 0.96, 0.97, 0.98, 0.99])
    ap.add_argument("--top-videos", type=int, default=15,
                    help="How many heaviest-drop videos to show per threshold")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Optional CSV summary path")
    ap.add_argument("--write-manifest-at", type=float, default=None,
                    help="Threshold for which to also emit a viewer-compatible "
                         "manifest.json + stem_to_path.json (selected = "
                         "survivors, dedup_drops = dropped, fps_drops = []).")
    ap.add_argument("--manifest-out-dir", type=Path, default=None,
                    help="Where to write the manifest pair. Required with "
                         "--write-manifest-at.")
    args = ap.parse_args()

    source_map: dict[str, dict] = json.loads(
        (args.working_dir / "source_map.json").read_text()
    )
    stems = sorted(source_map.keys())
    print(f"Loaded source_map with {len(stems)} stems "
          f"({Counter(m['source'] for m in source_map.values())})")

    print(f"Loading embeddings from {args.embeddings_dir} ...")
    embs, kept_stems = load_embeddings(stems, args.embeddings_dir)
    print(f"  shape: {embs.shape}  ({embs.nbytes / 1e6:.1f} MB)")

    print(f"Running threshold sweep over {len(args.thresholds)} values ...")
    per_thresh = greedy_dedup_threshold_sweep(embs, args.thresholds)

    # ----- print headline table -----
    print("\n" + "=" * 78)
    print(f"{'thr':>6}  {'survivors':>9} {'drop%':>6}  "
          f"{'AVA':>5}  {'animal_person':>14}  {'NWPU':>5}")
    print("-" * 78)
    rows = []
    for t in args.thresholds:
        a = analytics(per_thresh[t], kept_stems, source_map, args.top_videos)
        n = a["total_survivors"]
        drop_pct = 100.0 * (len(kept_stems) - n) / len(kept_stems)
        bs = a["by_source"]
        rows.append((t, n, drop_pct, bs))
        print(f"{t:>6.2f}  {n:>9} {drop_pct:>5.1f}%  "
              f"{bs.get('AVA', 0):>5}  {bs.get('animal_person', 0):>14}  "
              f"{bs.get('NWPU', 0):>5}")
    print("=" * 78)

    # ----- per-threshold heaviest-drop videos -----
    for t in args.thresholds:
        a = analytics(per_thresh[t], kept_stems, source_map, args.top_videos)
        if not a["ranked_videos"]:
            continue
        print(f"\n  threshold {t:.2f}  --  {args.top_videos} heaviest-drop videos:")
        print(f"    {'video':40} {'kept':>5} {'total':>5} {'kept%':>6}")
        for v in a["ranked_videos"]:
            kept = a["by_video_survivors"].get(v, 0)
            tot  = a["by_video_total"][v]
            pct  = 100.0 * kept / tot
            print(f"    {v:40} {kept:>5} {tot:>5} {pct:>5.1f}%")

    if args.write_manifest_at is not None:
        if args.manifest_out_dir is None:
            raise SystemExit("--manifest-out-dir is required with --write-manifest-at")
        if args.write_manifest_at not in per_thresh:
            raise SystemExit(
                f"--write-manifest-at {args.write_manifest_at} not in --thresholds; "
                "add it to the sweep so it's actually computed.")
        mask = per_thresh[args.write_manifest_at]
        selected = [s for s, alive in zip(kept_stems, mask) if alive]
        drops    = [s for s, alive in zip(kept_stems, mask) if not alive]
        # stem_to_path: from working_dir/images/<stem>.<ext>
        img_dir = args.working_dir / "images"
        s2p: dict[str, str] = {}
        ext_by_stem = {p.stem: p for p in img_dir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp")}
        for s in kept_stems:
            p = ext_by_stem.get(s)
            if p is not None:
                s2p[s] = str(p.resolve())

        manifest = {
            "input_dir": str(img_dir.resolve()),
            "embedding": {
                "model_id": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
                "stage":    "hidden_cls",
                "dim":      int(embs.shape[1]),
                "cache_dir": str(args.embeddings_dir.resolve()),
            },
            "dedup": {
                "threshold": args.write_manifest_at,
                "k_neighbors": None,
                "total":     len(kept_stems),
                "survivors": len(selected),
                "drops":     len(drops),
            },
            "selection": {
                "target":   len(selected),
                "selected": len(selected),
                "method":   "dedup_only_no_fps",
            },
            "selected":     selected,
            "dedup_drops":  drops,
            "fps_drops":    [],
        }
        out = args.manifest_out_dir
        out.mkdir(parents=True, exist_ok=True)
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (out / "stem_to_path.json").write_text(json.dumps(s2p, indent=2))
        print(f"\nWrote manifest at threshold {args.write_manifest_at}: {out}/manifest.json"
              f"  ({len(selected)} selected, {len(drops)} dropped)")

    if args.out_csv:
        import csv
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["threshold", "survivors", "drop_pct",
                        "AVA", "animal_person", "NWPU"])
            for t, n, dp, bs in rows:
                w.writerow([t, n, f"{dp:.2f}",
                            bs.get("AVA", 0), bs.get("animal_person", 0),
                            bs.get("NWPU", 0)])
        print(f"\nWrote summary CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
