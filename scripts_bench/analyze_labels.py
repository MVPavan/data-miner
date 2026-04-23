"""Exact class + confidence analysis over ALL YOLO label files written so far
by the laion_good2 SAM3-DART run. Safe to run while the job is still writing
(it snapshots sizes via a single scandir pass, then only reads non-empty files).
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path

LABELS_DIR = Path("/media/data_2/vlm/code/data_miner/output/auto_annotation_v4/laion_good2/labels_standalone")
CLASS_NAME = {16: "dog", 30: "shopping_cart", 33: "forklift", 34: "palletjack"}

t0 = time.perf_counter()
print(f"scanning {LABELS_DIR} ...", flush=True)

total = 0
empty = 0
non_empty_paths: list[str] = []
with os.scandir(LABELS_DIR) as it:
    for e in it:
        if not e.name.endswith(".txt"):
            continue
        total += 1
        try:
            size = e.stat(follow_symlinks=False).st_size
        except OSError:
            continue
        if size == 0:
            empty += 1
        else:
            non_empty_paths.append(e.path)

t_scan = time.perf_counter() - t0
print(f"scan: {total:,} files in {t_scan:.1f}s "
      f"({empty:,} empty, {len(non_empty_paths):,} non-empty)", flush=True)

det_per_class: Counter[int] = Counter()
files_per_class: Counter[int] = Counter()
conf_bucket: Counter[float] = Counter()  # 0.05 bins
total_dets = 0
low_conf_cap = int(0.25 * 100)

t1 = time.perf_counter()
for i, path in enumerate(non_empty_paths):
    try:
        with open(path, "r", encoding="utf-8") as f:
            classes_here: set[int] = set()
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    continue
                det_per_class[cid] += 1
                classes_here.add(cid)
                total_dets += 1
                if len(parts) >= 6:
                    try:
                        c = float(parts[5])
                    except ValueError:
                        continue
                    bucket = round(c * 20) / 20.0
                    conf_bucket[bucket] += 1
        for cid in classes_here:
            files_per_class[cid] += 1
    except OSError:
        continue
    if (i + 1) % 20000 == 0:
        print(f"  read {i+1:,}/{len(non_empty_paths):,}", flush=True)

t_read = time.perf_counter() - t1
print(f"read: {len(non_empty_paths):,} non-empty files in {t_read:.1f}s "
      f"({total_dets:,} detections)", flush=True)

print()
print("=" * 60)
print(f"LABEL ANALYSIS — laion_good2 (as of {time.strftime('%Y-%m-%d %H:%M:%S')})")
print("=" * 60)
print(f"total .txt files            : {total:>12,}")
print(f"  empty (no detections)     : {empty:>12,}  ({empty/total*100:.2f}%)")
print(f"  with >=1 detection        : {len(non_empty_paths):>12,}  ({len(non_empty_paths)/total*100:.2f}%)")
print(f"total detections            : {total_dets:>12,}")
print(f"detections per image (avg)  : {total_dets/total:>12.3f}")
print(f"detections per nonempty avg : {total_dets/max(1,len(non_empty_paths)):>12.3f}")
print()
print("Per-class (id  name             detections     imgs-with-class  %-of-nonempty)")
for cid in sorted(CLASS_NAME):
    name = CLASS_NAME[cid]
    n_det = det_per_class.get(cid, 0)
    n_img = files_per_class.get(cid, 0)
    pct = n_img / max(1, len(non_empty_paths)) * 100
    avg = n_det / max(1, n_img)
    print(f"  {cid:>3} {name:<16} {n_det:>12,}     {n_img:>12,}    {pct:>6.2f}%   "
          f"(avg {avg:.2f} per img)")
# unexpected class ids
other = sorted(k for k in det_per_class if k not in CLASS_NAME)
if other:
    print(f"\nUNEXPECTED class_ids (shouldn't happen with --classes filter):")
    for cid in other:
        print(f"  {cid}: {det_per_class[cid]:,} detections")

print()
print("Confidence distribution (0.05 bins, only for 6-col YOLO lines):")
if conf_bucket:
    scored = sum(conf_bucket.values())
    for b in sorted(conf_bucket):
        n = conf_bucket[b]
        bar = "#" * int(60 * n / max(conf_bucket.values()))
        print(f"  {b:.2f}  {n:>10,}  {n/scored*100:>5.2f}%  {bar}")
    # quick quantiles
    cum = 0
    cutoff = {}
    tot = scored
    sorted_buckets = sorted(conf_bucket)
    for b in sorted_buckets:
        cum += conf_bucket[b]
        for q in (0.5, 0.9, 0.99):
            if q not in cutoff and cum / tot >= q:
                cutoff[q] = b
    print(f"  approx conf quantiles : p50={cutoff.get(0.5,'?')}  "
          f"p90={cutoff.get(0.9,'?')}  p99={cutoff.get(0.99,'?')}")

print()
print(f"total wall: {time.perf_counter()-t0:.1f}s")
