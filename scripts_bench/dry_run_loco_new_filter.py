"""Dry-run the new LOCO filter rules on 10 sampled images.

No DB mutations — loads existing detect proposals + old filter output from
the production pipeline.db, simulates the new FilterPipeline with the
loco_rules.yaml override merged into the config, and prints a per-image diff.

Picks images to maximize coverage of the three new rules:
  - 3 images with ``head`` but NO ``person``           (exercises rule 1)
  - 3 images with ``forklift`` + ``palletjack``         (exercises rule 2 NMS)
  - 2 images with small head/cellphone/handbag candidates (exercises rule 3)
  - 2 random fallback images

Run:
    python scripts_bench/dry_run_loco_new_filter.py
"""
from __future__ import annotations

import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import yaml

# Allow running from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_miner.auto_annotation_v4.configs.contracts import Candidate
from data_miner.auto_annotation_v4.configs.enums import (
    CandidateStatus, FilterContext,
)
from data_miner.auto_annotation_v4.configs.settings import (
    AutoAnnotationV4Config,
)
from data_miner.auto_annotation_v4.filters import FilterPipeline
from data_miner.auto_annotation_v4.utils import route_candidates

DB = ROOT / "output/auto_annotation_v4/loco_unannotated_full_sam_filtered_detect/pipeline.db"
CFG_JSON = ROOT / "output/auto_annotation_v4/loco_unannotated_full_sam_filtered_detect/config.yaml"
OVERRIDE = ROOT / "data_miner/auto_annotation_v4/configs/overrides/loco_rules.yaml"


def load_config_with_override() -> AutoAnnotationV4Config:
    with open(CFG_JSON) as f:
        base = json.load(f)
    with open(OVERRIDE) as f:
        ov = yaml.safe_load(f)
    # Deep-merge override.filtering into base.filtering
    for k, v in ov.get("filtering", {}).items():
        if isinstance(v, dict) and isinstance(base["filtering"].get(k), dict):
            base["filtering"][k].update(v)
        else:
            base["filtering"][k] = v
    return AutoAnnotationV4Config(**base)


def candidates_from_json(data: list | dict) -> list[Candidate]:
    if isinstance(data, dict):
        cands_raw = data.get("candidates", [])
    else:
        cands_raw = data
    out = []
    for c in cands_raw:
        # Reuse existing status if present, else default
        status = c.get("status", CandidateStatus.PROPOSED.value)
        if isinstance(status, str):
            status = CandidateStatus(status)
        out.append(Candidate(
            candidate_id=c["candidate_id"],
            class_name=c["class_name"],
            label=c.get("label", c["class_name"]),
            source_model=c["source_model"],
            expression=c.get("expression", c["class_name"]),
            bbox=c["bbox"],
            score=c["score"],
            agreement=c.get("agreement", 1),
            agreeing_models=c.get("agreeing_models", [c["source_model"]]),
            status=status,
            mask_rle=c.get("mask_rle"),
            metadata=c.get("metadata", {}) or {},
            notes=c.get("notes", []) or [],
        ))
    return out


def pick_sample(con: sqlite3.Connection) -> list[str]:
    """Pick 10 images that cover the three new rules."""
    head_no_person: list[str] = []
    confusion: list[str] = []
    tiny: list[str] = []
    any_image: list[str] = []

    for iid, data in con.execute(
        "SELECT image_id, data FROM stages WHERE stage='filter'"
    ):
        d = json.loads(data)
        cls_set = {c["class_name"] for c in d.get("candidates", [])}
        any_image.append(iid)
        if "head" in cls_set and "person" not in cls_set:
            head_no_person.append(iid)
        if {"forklift", "palletjack"} <= cls_set or {"forklift", "shopping cart"} <= cls_set:
            confusion.append(iid)

    # Look into raw detect proposals for tiny candidates likely to benefit
    # from per_class_min_area override (to verify they'd be rescued).
    for iid, data in con.execute(
        "SELECT image_id, data FROM stages WHERE stage='detect'"
    ):
        d = json.loads(data)
        for c in d.get("candidates", []):
            if c["class_name"] not in {"head", "cellphone", "handbag", "bird"}:
                continue
            bb = c["bbox"]
            area = (bb["x2"] - bb["x1"]) * (bb["y2"] - bb["y1"])
            if 0.00005 <= area < 0.0005:
                tiny.append(iid)
                break
        if len(tiny) >= 30:
            break

    random.seed(42)
    sample: list[str] = []
    sample += random.sample(head_no_person, min(3, len(head_no_person)))
    sample += random.sample(confusion, min(3, len(confusion)))
    sample += random.sample([i for i in tiny if i not in sample], min(2, len(tiny)))
    while len(sample) < 10:
        extra = random.choice(any_image)
        if extra not in sample:
            sample.append(extra)
    return sample[:10]


def summarize(cands: list[Candidate]) -> str:
    if not cands:
        return "(empty)"
    by_class = Counter(c.class_name for c in cands)
    return ", ".join(f"{k}:{v}" for k, v in by_class.most_common())


def main() -> None:
    config = load_config_with_override()
    pipeline = FilterPipeline(config)

    con = sqlite3.connect(str(DB))
    con.row_factory = sqlite3.Row
    sample_ids = pick_sample(con)
    print(f"sampled {len(sample_ids)} images:")
    for iid in sample_ids:
        print(f"  {iid}")
    print()

    total_old_kept = 0
    total_new_kept = 0
    total_new_auto = 0
    total_new_vlm = 0
    drop_reason_counter: Counter = Counter()

    for iid in sample_ids:
        row = con.execute(
            "SELECT data FROM stages WHERE image_id=? AND stage='detect'",
            (iid,),
        ).fetchone()
        if row is None:
            print(f"[{iid}] SKIP: no detect stage in DB")
            continue
        detect_data = json.loads(row["data"])
        raw_cands = candidates_from_json(detect_data.get("candidates", []))

        # Old filter output (from DB)
        filt_row = con.execute(
            "SELECT data FROM stages WHERE image_id=? AND stage='filter'",
            (iid,),
        ).fetchone()
        old_cands = (
            candidates_from_json(json.loads(filt_row["data"]))
            if filt_row else []
        )

        # Simulate new filter
        new_cands, drops = pipeline.run(raw_cands, FilterContext.POST_DETECT)
        routing = route_candidates(new_cands, config)

        total_old_kept += len(old_cands)
        total_new_kept += len(new_cands)
        total_new_auto += len(routing["auto_accepted"])
        total_new_vlm += len(routing["needs_evaluation"])
        for d in drops:
            drop_reason_counter[d.reason.value] += 1

        print(f"--- {iid} ---")
        print(f"  raw       : {len(raw_cands):3d}  {summarize(raw_cands)}")
        print(f"  old filter: {len(old_cands):3d}  {summarize(old_cands)}")
        print(f"  new filter: {len(new_cands):3d}  {summarize(new_cands)}")
        if drops:
            reason_tally: Counter = Counter(d.reason.value for d in drops)
            print(f"  new drops : {dict(reason_tally)}")
        print(f"  new route : auto_accept={len(routing['auto_accepted'])}, "
              f"needs_vlm={len(routing['needs_evaluation'])}, "
              f"confusion_flags={len(routing['confusion_flags'])}")
        print()

    print("=" * 60)
    print("SUMMARY (10 images)")
    print("=" * 60)
    print(f"  old filter kept : {total_old_kept}")
    print(f"  new filter kept : {total_new_kept}")
    print(f"  new auto-accept : {total_new_auto}")
    print(f"  new needs_vlm   : {total_new_vlm}")
    print(f"  new drops by reason:")
    for r, n in drop_reason_counter.most_common():
        print(f"     {r:20s} {n:5d}")


if __name__ == "__main__":
    main()
