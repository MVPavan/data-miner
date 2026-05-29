"""Standalone batch-size sweep harness for v4 GDINO and SAM3-DART models.

Sweeps (B, N) on the actual hardware to pick real values for a 1M-image job.
For each (model, B, N) combo: warmup once, time 3 infer_batch calls, take median;
record peak GPU memory + throughput. OOMs are caught and logged, not fatal.

Run:
    python -m data_miner.auto_annotation_v4.tests.bench_batch_sizes
    python -m data_miner.auto_annotation_v4.tests.bench_batch_sizes --model gdino
    python -m data_miner.auto_annotation_v4.tests.bench_batch_sizes --model sam3_dart
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from PIL import Image

REPO    = Path("/media/data_2/vlm/code/data_miner")
SAMPLES = REPO / "output" / "sample" / "fl_pj_sample"

CLASSES_20 = [
    "person", "forklift", "pallet jack", "box",
    "cart", "truck", "door", "shelf",
    "conveyor", "ladder", "sign", "pallet",
    "helmet", "worker", "crate", "barrel",
    "rack", "bin", "tape", "wire",
]

PROMPT_SETS = {
    4:  CLASSES_20[:4],
    8:  CLASSES_20[:8],
    16: CLASSES_20[:16],
    20: CLASSES_20[:20],
}

N_VALUES = [4, 8, 16, 20]
B_VALUES = [4, 8, 16, 32]

N_TIMED   = 3
GDINO_THR = 0.25
SAM3_THR  = 0.5


def _load_images() -> list[Image.Image]:
    paths = sorted(SAMPLES.glob("*.jpg"))
    assert paths, f"No JPEGs found in {SAMPLES}"
    return [Image.open(p).convert("RGB") for p in paths]


def _get_images(n: int) -> list[Image.Image]:
    base = _load_images()
    reps = math.ceil(n / len(base))
    return (base * reps)[:n]


def _reset_cuda() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class CellResult:
    __slots__ = ("status", "mem_gb", "lat_s", "tput")

    def __init__(self, status: str, mem_gb: float = 0.0,
                 lat_s: float = 0.0, tput: float = 0.0) -> None:
        self.status = status
        self.mem_gb = mem_gb
        self.lat_s  = lat_s
        self.tput   = tput

    def fmt(self) -> str:
        if self.status != "OK":
            return f"{self.status:^22}"
        return f"m={self.mem_gb:4.1f}G l={self.lat_s:5.2f}s t={self.tput:5.1f}"


def measure(model, B: int, N: int, threshold: float) -> CellResult:
    import torch

    _reset_cuda()
    prompts = PROMPT_SETS[N]
    try:
        imgs = _get_images(B)
        prepared = [model.prepare(img, prompts, threshold=threshold) for img in imgs]

        # Warmup: first call pays one-time alloc + autotune cost.
        _ = model.infer_batch(prepared)
        _sync()

        torch.cuda.reset_peak_memory_stats()
        latencies: list[float] = []
        for _ in range(N_TIMED):
            _sync()
            t0 = time.perf_counter()
            _ = model.infer_batch(prepared)
            _sync()
            latencies.append(time.perf_counter() - t0)

        peak_bytes = torch.cuda.max_memory_allocated()
        med = statistics.median(latencies)
        return CellResult("OK",
                          mem_gb=peak_bytes / 1e9,
                          lat_s=med,
                          tput=B / med if med > 0 else 0.0)
    except torch.cuda.OutOfMemoryError:
        _reset_cuda()
        return CellResult("OOM")
    except Exception as exc:
        _reset_cuda()
        msg = str(exc).splitlines()[0][:18] if str(exc) else type(exc).__name__
        return CellResult(f"ERR:{msg}")


def sweep(model, name: str, threshold: float) -> dict[tuple[int, int], CellResult]:
    print(f"\n=== Sweeping {name} (threshold={threshold}) ===")
    results: dict[tuple[int, int], CellResult] = {}
    for N in N_VALUES:
        for B in B_VALUES:
            print(f"  [{name}] B={B:>3} N={N:>3} ... ", end="", flush=True)
            r = measure(model, B, N, threshold)
            results[(N, B)] = r
            print(r.fmt())
    return results


def print_table(name: str, results: dict[tuple[int, int], CellResult]) -> None:
    print(f"\n{name}")
    header = f"    {'N\\B':<6}" + "".join(f" {b:>22}" for b in B_VALUES)
    print(header)
    print("    " + "-" * (len(header) - 4))
    for N in N_VALUES:
        row = f"    {N:<6}"
        for B in B_VALUES:
            r = results.get((N, B))
            cell = r.fmt() if r else " " * 22
            row += f" {cell:>22}"
        print(row)


def recommend(name: str, results: dict[tuple[int, int], CellResult]) -> None:
    ok = [((N, B), r) for (N, B), r in results.items() if r.status == "OK"]
    if not ok:
        print(f"  [{name}] no successful cells — all OOM/ERR")
        return
    best = max(ok, key=lambda kv: kv[1].tput / max(kv[1].mem_gb, 0.1))
    (N, B), r = best
    best_tput = max(ok, key=lambda kv: kv[1].tput)
    (Nt, Bt), rt = best_tput
    print(f"  [{name}] best tput/GB: B={B} N={N} -> "
          f"{r.tput:.1f} img/s @ {r.mem_gb:.1f}GB (eff={r.tput/r.mem_gb:.2f})")
    print(f"  [{name}] peak tput  : B={Bt} N={Nt} -> "
          f"{rt.tput:.1f} img/s @ {rt.mem_gb:.1f}GB")


def run_gdino() -> dict[tuple[int, int], CellResult]:
    from data_miner.auto_annotation_v4.models.grounding_dino import GDINOModel
    print("Loading GDINOModel ...")
    m = GDINOModel()
    m.load(device="cuda:0", model_id="IDEA-Research/grounding-dino-base")
    try:
        return sweep(m, "GDINO", GDINO_THR)
    finally:
        del m
        _reset_cuda()


def run_sam3() -> dict[tuple[int, int], CellResult]:
    from data_miner.auto_annotation_v4.models.sam3_dart import SAM3DartModel
    print("Loading SAM3DartModel ...")
    m = SAM3DartModel()
    m.load(device="cuda:0", model_id="sam3_dart",
           detection_only=True, presence_threshold=0.05)
    try:
        return sweep(m, "SAM3-DART", SAM3_THR)
    finally:
        del m
        _reset_cuda()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-size sweep for v4 detectors.")
    parser.add_argument("--model", choices=["gdino", "sam3_dart", "both"],
                        default="both", help="Which model to sweep (default: both)")
    args = parser.parse_args()

    tables: list[tuple[str, dict[tuple[int, int], CellResult]]] = []

    if args.model in ("gdino", "both"):
        try:
            tables.append(("GDINO", run_gdino()))
        except Exception:
            traceback.print_exc()

    if args.model in ("sam3_dart", "both"):
        try:
            tables.append(("SAM3-DART", run_sam3()))
        except Exception:
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for name, results in tables:
        print_table(name, results)

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    for name, results in tables:
        recommend(name, results)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
