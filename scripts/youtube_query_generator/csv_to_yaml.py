#!/usr/bin/env python3
import csv
import json
from pathlib import Path

CSV_PATH = Path("scripts/youtube_query_generator/output/youtube_queries.csv")
OUT_PATH = Path("run_configs/youtube_queries_converted.yaml")

terms = []
with CSV_PATH.open(newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader, None)
    for row in reader:
        if not row:
            continue
        term = row[0].strip()
        if term:
            terms.append(term)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUT_PATH.open("w", encoding='utf-8') as f:
    f.write("queries:\n")
    for t in terms:
        # Use JSON to produce a safely-escaped double-quoted string, allow unicode
        f.write("  - " + json.dumps(t, ensure_ascii=False) + "\n")

print(f"Wrote {len(terms)} queries to {OUT_PATH}")
