"""
Tiny FastAPI viewer for a select_diverse_subset.py manifest.

Serves a single page with three tabs (selected / dedup_drops / fps_drops),
each rendering a paginated thumbnail grid. Lets you eyeball whether the FPS
selection actually retained the diversity it claimed and whether the dedup
drops were genuine duplicates.

Usage:
    python -m scripts.dataset_selection.viewer_subset \
        --manifest output/dataset_selection/datatang_diverse_1000/manifest.json \
        --port 8765
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse


_CLIP_FRAME_RE = re.compile(r"^(?P<clip>.+?)_f\d+$")
_CLIP_NUM_RE = re.compile(r"^(?P<clip>.+?)_\d+$")
_DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")


def _strip_one(stem: str) -> str:
    """Strip one trailing _f<digits> or _<digits>; return unchanged if neither."""
    m = _CLIP_FRAME_RE.match(stem)
    if m:
        return m.group("clip")
    m = _CLIP_NUM_RE.match(stem)
    if m:
        return m.group("clip")
    return stem


def derive_clip_map(stems: list[str]) -> dict[str, str]:
    """Build stem → clip mapping using a two-pass count-based heuristic.

    Pass 1: candidate = strip one _f<digits> or _<digits>.
            Frequent (>=2) candidates are accepted as real clip names.
    Pass 2: singleton candidates whose strip-one parent is a frequent clip
            get reassigned to the parent (collapses sub-indexed frames like
            `1_1_0105_1` back to `1_1`).
    Pass 3: any remaining singleton matching ^YYYY-MM-DD gets bucketed under
            its date — DataTang has time-lapse stems like `2018-10-09-06:31:42`
            that should share one group.
    """
    from collections import Counter

    cand1 = [_strip_one(s) for s in stems]
    counts = Counter(cand1)
    real = {c for c, n in counts.items() if n >= 2}

    out: dict[str, str] = {}
    for stem, c1 in zip(stems, cand1):
        if counts[c1] >= 2:
            out[stem] = c1
            continue
        c2 = _strip_one(c1)
        if c2 != c1 and c2 in real:
            out[stem] = c2
            continue
        m = _DATE_PREFIX_RE.match(c1)
        if m:
            out[stem] = m.group(1)
            continue
        out[stem] = c1
    return out


def derive_clip_map_with_overrides(
    stems: list[str],
    canonical: list[str] | None,
) -> dict[str, str]:
    """Use canonical clip names (longest-prefix wins) when provided;
    otherwise fall back to derive_clip_map.
    """
    if not canonical:
        return derive_clip_map(stems)

    # Sort by length desc so longest prefix wins; tie-break alphabetic.
    ordered = sorted(set(canonical), key=lambda c: (-len(c), c))
    out: dict[str, str] = {}
    fallback = derive_clip_map(stems)
    for stem in stems:
        match = None
        for clip in ordered:
            if stem == clip or stem.startswith(clip + "_"):
                match = clip
                break
        out[stem] = match if match is not None else fallback[stem]
    return out


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Diverse-subset viewer</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; background: #111; color: #eee; }
  header { padding: 12px 20px; background: #1c1c1c; display: flex; gap: 18px;
           align-items: baseline; border-bottom: 1px solid #333; }
  header h1 { font-size: 16px; margin: 0; font-weight: 500; }
  header .stats { color: #888; font-size: 13px; }
  nav { display: flex; gap: 4px; padding: 8px 20px; background: #181818;
        border-bottom: 1px solid #333; }
  nav button { background: #2a2a2a; color: #ddd; border: 1px solid #444;
               padding: 6px 14px; cursor: pointer; font-size: 13px; }
  nav button.active { background: #4a7; color: #111; border-color: #4a7; }
  .layout { display: flex; min-height: calc(100vh - 95px); }
  aside { width: 240px; background: #181818; border-right: 1px solid #333;
          overflow-y: auto; max-height: calc(100vh - 95px); padding: 8px 0; }
  aside .clip-row { padding: 5px 12px; cursor: pointer; font-size: 12px;
                    color: #ccc; display: flex; justify-content: space-between;
                    gap: 8px; font-family: ui-monospace, monospace; }
  aside .clip-row:hover { background: #2a2a2a; }
  aside .clip-row.active { background: #2a4; color: #111; }
  aside .clip-row .n { color: #777; }
  aside .clip-row.active .n { color: #333; }
  aside h3 { font-size: 11px; text-transform: uppercase; color: #888;
             padding: 4px 12px; margin: 0 0 4px; letter-spacing: 0.05em; }
  main { flex: 1; min-width: 0; }
  .grid { display: grid; gap: 4px; padding: 12px;
          grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); }
  .group-header { grid-column: 1 / -1; padding: 10px 4px 4px;
                  font-size: 13px; color: #9cf; border-bottom: 1px solid #333;
                  margin-top: 6px; font-family: ui-monospace, monospace; }
  .group-header .count { color: #666; margin-left: 8px; font-size: 11px; }
  .cell { position: relative; aspect-ratio: 4/3; background: #000;
          overflow: hidden; }
  .cell img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .cell .label { position: absolute; left: 0; bottom: 0; right: 0;
                 padding: 2px 6px; background: rgba(0,0,0,0.7);
                 font-size: 10px; color: #ccc;
                 white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .pager { padding: 12px 20px; display: flex; gap: 12px; align-items: center; }
  .pager button { background: #2a2a2a; color: #ddd; border: 1px solid #444;
                  padding: 4px 12px; cursor: pointer; }
  .pager button:disabled { opacity: 0.4; cursor: default; }
</style>
</head>
<body>
<header>
  <h1>Diverse-subset viewer</h1>
  <span id="stats" class="stats">loading…</span>
</header>
<nav>
  <button data-tab="selected" class="active">Selected</button>
  <button data-tab="dedup_drops">Dedup drops</button>
  <button data-tab="fps_drops">FPS drops</button>
</nav>
<div class="layout">
  <aside>
    <h3 id="clipindex-title">clips</h3>
    <div id="clipindex"></div>
  </aside>
  <main>
    <div class="pager">
      <button id="prev">← prev</button>
      <span id="pageinfo"></span>
      <button id="next">next →</button>
      <button id="clear-clip" style="display:none;">× clear clip filter</button>
    </div>
    <div id="grid" class="grid"></div>
  </main>
</div>

<script>
const PAGE_SIZE = 120;
let state = { tab: "selected", page: 0, total: 0, clipFilter: null };

async function loadManifestStats() {
  const r = await fetch("/api/stats");
  const s = await r.json();
  document.getElementById("stats").textContent =
    `total ${s.total} • selected ${s.selected} • dedup drop ${s.dedup_drops} • fps drop ${s.fps_drops}`;
}

async function loadClipIndex() {
  const r = await fetch(`/api/clips?bucket=${state.tab}`);
  const data = await r.json();
  document.getElementById("clipindex-title").textContent =
    `clips (${data.total_clips}) — ${state.tab}`;
  const box = document.getElementById("clipindex");
  box.innerHTML = "";
  for (const {clip, count} of data.clips) {
    const row = document.createElement("div");
    row.className = "clip-row" + (state.clipFilter === clip ? " active" : "");
    row.dataset.clip = clip;
    const name = document.createElement("span");
    name.textContent = clip;
    name.title = clip;
    name.style.overflow = "hidden";
    name.style.textOverflow = "ellipsis";
    name.style.whiteSpace = "nowrap";
    const n = document.createElement("span");
    n.className = "n";
    n.textContent = count;
    row.appendChild(name);
    row.appendChild(n);
    row.onclick = () => {
      state.clipFilter = (state.clipFilter === clip) ? null : clip;
      state.page = 0;
      document.getElementById("clear-clip").style.display =
        state.clipFilter ? "inline-block" : "none";
      loadClipIndex();
      loadPage();
    };
    box.appendChild(row);
  }
}

async function loadPage() {
  const params = new URLSearchParams({
    bucket: state.tab,
    offset: state.page * PAGE_SIZE,
    limit: PAGE_SIZE,
  });
  if (state.clipFilter) params.set("clip", state.clipFilter);
  const r = await fetch(`/api/list?${params}`);
  const data = await r.json();
  state.total = data.total;
  const grid = document.getElementById("grid");
  grid.innerHTML = "";

  // Group items by clip in render order; emit a header whenever the clip
  // differs from the prior cell (or from the previous page's last clip).
  let lastClip = data.prev_clip;
  let groupCount = 0;
  let headerEl = null;
  const flushHeader = () => {
    if (headerEl) headerEl.querySelector(".count").textContent = `(${groupCount})`;
  };

  for (const it of data.items) {
    if (it.clip !== lastClip) {
      flushHeader();
      headerEl = document.createElement("div");
      headerEl.className = "group-header";
      const name = document.createElement("span");
      name.textContent = it.clip;
      const count = document.createElement("span");
      count.className = "count";
      headerEl.appendChild(name);
      headerEl.appendChild(count);
      grid.appendChild(headerEl);
      lastClip = it.clip;
      groupCount = 0;
    }
    groupCount++;

    const cell = document.createElement("div");
    cell.className = "cell";
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = `/api/image/${encodeURIComponent(it.stem)}`;
    const lab = document.createElement("div");
    lab.className = "label";
    lab.textContent = it.stem;
    cell.appendChild(img);
    cell.appendChild(lab);
    grid.appendChild(cell);
  }
  flushHeader();
  const pages = Math.max(1, Math.ceil(state.total / PAGE_SIZE));
  document.getElementById("pageinfo").textContent =
    `${state.tab}: ${state.page + 1} / ${pages}  (${state.total} items)`;
  document.getElementById("prev").disabled = state.page === 0;
  document.getElementById("next").disabled = state.page + 1 >= pages;
}

document.querySelectorAll("nav button").forEach(b => {
  if (!b.dataset.tab) return;
  b.onclick = () => {
    document.querySelectorAll("nav button[data-tab]").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    state.tab = b.dataset.tab;
    state.page = 0;
    state.clipFilter = null;
    document.getElementById("clear-clip").style.display = "none";
    loadClipIndex();
    loadPage();
  };
});
document.getElementById("prev").onclick = () => { state.page--; loadPage(); };
document.getElementById("next").onclick = () => { state.page++; loadPage(); };
document.getElementById("clear-clip").onclick = () => {
  state.clipFilter = null; state.page = 0;
  document.getElementById("clear-clip").style.display = "none";
  loadClipIndex(); loadPage();
};

loadManifestStats();
loadClipIndex();
loadPage();
</script>
</body>
</html>
"""


def build_app(
    manifest_path: Path,
    clips_file: Path | None = None,
) -> FastAPI:
    manifest = json.loads(manifest_path.read_text())
    stem_to_path_file = manifest_path.parent / "stem_to_path.json"
    stem_to_path: dict[str, str] = json.loads(stem_to_path_file.read_text())

    canonical: list[str] | None = None
    if clips_file is not None:
        canonical = [
            line.strip()
            for line in clips_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    all_stems = manifest["selected"] + manifest["dedup_drops"] + manifest["fps_drops"]
    stem_to_clip = derive_clip_map_with_overrides(all_stems, canonical)

    # Sort each bucket by (clip, stem) so video frames cluster together and
    # within a clip frames go in stem order.
    def _sorted(stems: list[str]) -> list[str]:
        return sorted(stems, key=lambda s: (stem_to_clip[s], s))

    buckets = {
        "selected": _sorted(manifest["selected"]),
        "dedup_drops": _sorted(manifest["dedup_drops"]),
        "fps_drops": _sorted(manifest["fps_drops"]),
    }

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/stats")
    def stats() -> dict:
        return {
            "total": manifest["dedup"]["total"],
            "selected": len(buckets["selected"]),
            "dedup_drops": len(buckets["dedup_drops"]),
            "fps_drops": len(buckets["fps_drops"]),
            "manifest": str(manifest_path),
        }

    @app.get("/api/clips")
    def clips_index(
        bucket: str = Query(..., regex="^(selected|dedup_drops|fps_drops)$"),
    ) -> dict:
        """Return per-clip counts for the active bucket, sorted by count desc."""
        from collections import Counter
        clip_counts = Counter(stem_to_clip[s] for s in buckets[bucket])
        return {
            "bucket": bucket,
            "total_clips": len(clip_counts),
            "clips": [
                {"clip": c, "count": n}
                for c, n in sorted(clip_counts.items(), key=lambda x: (-x[1], x[0]))
            ],
        }

    @app.get("/api/list")
    def list_bucket(
        bucket: str = Query(..., regex="^(selected|dedup_drops|fps_drops)$"),
        offset: int = 0,
        limit: int = 120,
        clip: str | None = None,
    ) -> dict:
        items = buckets[bucket]
        if clip is not None:
            items = [s for s in items if stem_to_clip[s] == clip]
        page = items[offset : offset + limit]
        prev_clip = stem_to_clip[items[offset - 1]] if offset > 0 else None
        return {
            "bucket": bucket,
            "total": len(items),
            "offset": offset,
            "clip_filter": clip,
            "prev_clip": prev_clip,
            "items": [{"stem": s, "clip": stem_to_clip[s]} for s in page],
        }

    @app.get("/api/image/{stem}")
    def image(stem: str) -> FileResponse:
        path = stem_to_path.get(stem)
        if path is None or not Path(path).exists():
            raise HTTPException(404, f"stem not found: {stem}")
        return FileResponse(path)

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="Path to manifest.json from select_diverse_subset.py")
    ap.add_argument("--clips-file", type=Path, default=None,
                    help="Optional newline-delimited list of canonical video "
                         "names. Stems are grouped by longest-prefix match "
                         "(stem == clip or stem starts with '<clip>_'). "
                         "Falls back to the auto-derived heuristic for stems "
                         "that don't match any name.")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    app = build_app(args.manifest, clips_file=args.clips_file)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
