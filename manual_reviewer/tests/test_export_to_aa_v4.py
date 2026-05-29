"""Tests for the export script's LS fetch path.

Live regression: LS's ``/api/projects/{id}/export?exportType=JSON`` returns
``predictions: [<int_id>]`` (just IDs, not full prediction objects), which
would silently drop the seeded predictions and downgrade every kept region
to ``source="added"`` instead of ``finalize`` / ``relabeled`` / ``edited``.

The current ``_fetch_from_ls`` walks ``/api/projects/{id}/tasks`` (which
inlines full predictions) instead. These tests pin both the request shape
the script issues and the parser's behaviour on real-shaped seed +
annotation payloads.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from manual_reviewer.scripts import export_to_aa_v4 as mod


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self) -> Any:
        return self._payload


class _FakeClient:
    """Stand-in for httpx.Client. Records calls so tests can assert on them."""

    def __init__(self, pages: list[Any]):
        self._pages = list(pages)
        self.calls: list[dict[str, Any]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url: str, params: dict[str, Any] | None = None) -> _FakeResponse:
        self.calls.append({"url": url, "params": dict(params or {})})
        if not self._pages:
            return _FakeResponse([])
        return _FakeResponse(self._pages.pop(0))


@pytest.fixture
def patch_httpx(monkeypatch: pytest.MonkeyPatch):
    """Replace ``httpx.Client`` with a queue-driven fake."""

    holder: dict[str, _FakeClient] = {}

    def _install(pages: list[Any]) -> _FakeClient:
        fake = _FakeClient(pages)

        class _FakeHttpxModule:
            Client = lambda *a, **kw: fake  # noqa: E731

        monkeypatch.setitem(__import__("sys").modules, "httpx", _FakeHttpxModule)
        holder["fake"] = fake
        return fake

    return _install


def _seeded_region(
    *,
    region_id: str,
    cls: str,
    x: float = 10.0,
    y: float = 20.0,
    w: float = 30.0,
    h: float = 40.0,
) -> dict[str, Any]:
    return {
        "id": region_id,
        "type": "rectanglelabels",
        "from_name": "bbox",
        "to_name": "image",
        "origin": "prediction",
        "value": {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "rotation": 0,
            "rectanglelabels": [cls],
        },
    }


def _annotation_region(*, region_id: str, cls: str, x: float = 10.0,
                       y: float = 20.0, w: float = 30.0, h: float = 40.0,
                       origin: str = "prediction") -> dict[str, Any]:
    return {
        "id": region_id,
        "type": "rectanglelabels",
        "from_name": "bbox",
        "to_name": "image",
        "origin": origin,
        "value": {
            "x": x, "y": y, "width": w, "height": h,
            "rotation": 0, "rectanglelabels": [cls],
        },
    }


def _make_task(image_id: str, *, seeds: list[dict[str, Any]],
               regions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": 1,
        "data": {"image_id": image_id},
        "predictions": [{"id": 100, "result": list(seeds)}],
        "annotations": [
            {
                "id": 7,
                "completed_by": 1,
                "lead_time": 12.5,
                "result": list(regions),
            }
        ],
    }


def _args(**overrides: Any) -> argparse.Namespace:
    base = dict(
        ls_url="http://localhost:8080",
        ls_token="t",
        ls_project=1,
        ls_timeout=10.0,
        since=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# tests: request shape
# ---------------------------------------------------------------------------


def test_fetch_uses_tasks_endpoint_not_export(patch_httpx) -> None:
    """The export endpoint truncates predictions to IDs; tasks endpoint inlines them."""
    fake = patch_httpx([[]])  # one empty page → loop exits cleanly
    mod._fetch_from_ls(_args())
    assert fake.calls, "expected at least one HTTP call"
    first = fake.calls[0]
    assert first["url"].endswith("/api/projects/1/tasks"), (
        f"expected /tasks endpoint, got {first['url']!r} — using /export would "
        "regress the predictions truncation bug"
    )
    assert first["params"].get("page") == 1
    assert first["params"].get("page_size")  # paging is required


def test_fetch_pages_through_results(patch_httpx) -> None:
    page1 = [
        _make_task("img_a", seeds=[_seeded_region(region_id="r1", cls="truck")],
                   regions=[_annotation_region(region_id="r1", cls="truck")])
        for _ in range(100)
    ]
    page2 = [
        _make_task("img_b", seeds=[_seeded_region(region_id="r2", cls="car")],
                   regions=[_annotation_region(region_id="r2", cls="car")])
    ]
    fake = patch_httpx([page1, page2, []])
    out = mod._fetch_from_ls(_args())
    pages_seen = [c["params"]["page"] for c in fake.calls]
    assert pages_seen == [1, 2], f"expected to page until short page, got {pages_seen}"
    # 100 from page1 + 1 from page2 → 101 (annotation, data, seeded) tuples
    assert len(out) == 101


def test_fetch_skips_tasks_without_annotations(patch_httpx) -> None:
    page = [
        {"id": 1, "data": {"image_id": "no_ann"}, "annotations": []},
        _make_task("with_ann", seeds=[_seeded_region(region_id="r", cls="dog")],
                   regions=[_annotation_region(region_id="r", cls="dog")]),
    ]
    patch_httpx([page, []])
    out = mod._fetch_from_ls(_args())
    image_ids = [data.get("image_id") for _ann, data, _seed in out]
    assert image_ids == ["with_ann"]


# ---------------------------------------------------------------------------
# tests: classification round-trip on the parser
# ---------------------------------------------------------------------------


def test_parser_classifies_finalize_relabeled_edited_added(patch_httpx) -> None:
    """The point of preserving full predictions: parse_ls_completion needs them
    to distinguish ``finalize`` (untouched seed) from ``relabeled`` / ``edited``
    / ``added``. Walks the same tuple shape ``main()`` consumes.
    """
    seeds = [
        _seeded_region(region_id="kept",      cls="truck"),
        _seeded_region(region_id="relabeled", cls="motorcycle", x=50.0, y=50.0),
        _seeded_region(region_id="edited",    cls="person",     x=10.0, y=10.0,
                       w=20.0, h=20.0),
        _seeded_region(region_id="deleted",   cls="head",       x=70.0, y=70.0),
    ]
    regions = [
        _annotation_region(region_id="kept",      cls="truck"),
        _annotation_region(region_id="relabeled", cls="bicycle", x=50.0, y=50.0),
        _annotation_region(region_id="edited",    cls="person",  x=15.0, y=15.0,
                           w=25.0, h=25.0),
        _annotation_region(region_id="brand_new", cls="dog", x=5.0, y=5.0,
                           w=10.0, h=10.0, origin="manual"),
    ]
    page = [_make_task("img1", seeds=seeds, regions=regions)]
    patch_httpx([page, []])

    fetched = mod._fetch_from_ls(_args())
    assert len(fetched) == 1
    completion, task_data, predictions = fetched[0]
    assert task_data["image_id"] == "img1"
    seeded_regions = mod._extract_seeded(predictions)
    assert len(seeded_regions) == 4, (
        "seeded predictions[0].result should round-trip end-to-end — "
        "the bug fix's whole point"
    )

    from manual_reviewer.pipeline_io import parse_ls_completion

    result = parse_ls_completion(
        completion,
        image_id="img1",
        seeded_predictions=seeded_regions,
    )
    by_id = {c.candidate_id: c for c in result.corrections}
    assert by_id["kept"].source == "finalize"
    assert by_id["relabeled"].source == "relabeled"
    assert by_id["relabeled"].original_class == "motorcycle"
    assert by_id["edited"].source == "edited"
    assert by_id["edited"].original_bbox is not None
    assert by_id["brand_new"].source == "added"
    assert "deleted" in result.deletions


def test_parser_can_return_exchange_result(patch_httpx) -> None:
    """LS parsing should expose the neutral exchange model without changing v4 output."""
    seeds = [_seeded_region(region_id="kept", cls="truck")]
    regions = [_annotation_region(region_id="kept", cls="truck")]
    page = [_make_task("img1", seeds=seeds, regions=regions)]
    patch_httpx([page, []])
    completion, task_data, predictions = mod._fetch_from_ls(_args())[0]

    from data_miner.annotation_io import FrontendName, ReviewRegionOrigin
    from manual_reviewer.pipeline_io import parse_ls_completion_to_exchange_result

    exchange = parse_ls_completion_to_exchange_result(
        completion,
        image_id=task_data["image_id"],
        seeded_predictions=mod._extract_seeded(predictions),
    )

    assert exchange.source_frontend is FrontendName.LABEL_STUDIO
    assert exchange.source_task_id == "1"
    assert exchange.source_completion_id == "7"
    assert exchange.boxes[0].origin is ReviewRegionOrigin.PREDICTION
    assert exchange.to_human_review_result().corrections[0].source == "finalize"


def test_fetch_filters_by_since(patch_httpx, monkeypatch: pytest.MonkeyPatch) -> None:
    page = [
        _make_task("old", seeds=[_seeded_region(region_id="r", cls="dog")],
                   regions=[_annotation_region(region_id="r", cls="dog")]),
        _make_task("new", seeds=[_seeded_region(region_id="r", cls="cat")],
                   regions=[_annotation_region(region_id="r", cls="cat")]),
    ]
    page[0]["annotations"][0]["updated_at"] = "2026-01-01T00:00:00Z"
    page[1]["annotations"][0]["updated_at"] = "2026-06-01T00:00:00Z"
    patch_httpx([page, []])

    # Cutoff between the two annotation timestamps → only ``new`` passes.
    from datetime import datetime, timezone
    cutoff = datetime(2026, 3, 1, tzinfo=timezone.utc).timestamp()
    out = mod._fetch_from_ls(_args(since=cutoff))
    image_ids = [data.get("image_id") for _ann, data, _seed in out]
    assert image_ids == ["new"]


# ---------------------------------------------------------------------------
# argparse hardening
# ---------------------------------------------------------------------------


def test_argparse_requires_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LS_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        mod._parse_args(["--db", "/tmp/x.db"])


def test_argparse_in_file_with_since_rejected(tmp_path) -> None:
    f = tmp_path / "x.json"
    f.write_text("[]")
    with pytest.raises(SystemExit):
        mod._parse_args(["--db", "/tmp/x.db", "--in-file", str(f), "--since", "1.0"])


def test_argparse_ls_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    monkeypatch.setenv("LS_TOKEN", "env_tok")
    importlib.reload(mod)
    args = mod._parse_args(
        ["--db", "/tmp/x.db", "--ls-url", "http://x", "--ls-project", "1"]
    )
    assert args.ls_token == "env_tok"


# ---------------------------------------------------------------------------
# token redaction in error logs
# ---------------------------------------------------------------------------


def test_fetch_redacts_token_in_error(patch_httpx) -> None:
    fake = patch_httpx([])
    # Override get to return 5xx with token in body
    def _bad_get(url, params=None):
        fake.calls.append({"url": url, "params": dict(params or {})})
        r = _FakeResponse([], status_code=500)
        r.text = "internal error: token=secret_t in headers"
        return r
    fake.get = _bad_get  # type: ignore[assignment]
    with pytest.raises(RuntimeError) as ei:
        mod._fetch_from_ls(_args(ls_token="secret_t"))
    assert "secret_t" not in str(ei.value)
    assert "<redacted>" in str(ei.value)


# ---------------------------------------------------------------------------
# classes_file fail-fast
# ---------------------------------------------------------------------------


def test_rewrite_yolo_empty_classes_file_raises(tmp_path) -> None:
    from manual_reviewer.scripts.export_to_aa_v4 import _rewrite_yolo_label
    classes = tmp_path / "classes.txt"
    classes.write_text("\n  \n\n", encoding="utf-8")
    labels_dir = tmp_path / "labels"

    class _Bbox:
        x1, y1, x2, y2 = 0.1, 0.1, 0.2, 0.2

    class _C:
        class_name = "anything"
        bbox = _Bbox()

    class _Result:
        corrections = [_C()]
        frame_state = "clean"

    with pytest.raises(ValueError, match="empty or malformed"):
        _rewrite_yolo_label(labels_dir, "img1", _Result(), classes)


def test_trace_dedup_fallback_uses_reviewed_at(tmp_path) -> None:
    """When ls_completion_id is 0/falsy, dedup falls back to (image_id, reviewed_at)
    so re-running export doesn't duplicate the trace block."""
    from data_miner.auto_annotation_v4.configs.contracts import HumanReviewResult

    traces_dir = tmp_path / "traces"
    result = HumanReviewResult(
        image_id="img_a",
        reviewer_id="r",
        reviewed_at=42.0,
        ls_completion_id=0,
    )
    mod._append_trace(traces_dir, "img_a", result)
    mod._append_trace(traces_dir, "img_a", result)

    import json
    payload = json.loads((traces_dir / "img_a.json").read_text())
    assert isinstance(payload, list)
    assert len(payload) == 1, "duplicate trace blocks were not dedup'd"


def test_trace_refuses_when_no_id_and_no_reviewed_at(tmp_path, caplog) -> None:
    from data_miner.auto_annotation_v4.configs.contracts import HumanReviewResult

    traces_dir = tmp_path / "traces"
    result = HumanReviewResult(
        image_id="img_a", reviewer_id="r", reviewed_at=0.0,
        ls_completion_id=0,
    )
    with caplog.at_level("WARNING"):
        mod._append_trace(traces_dir, "img_a", result)
    assert not (traces_dir / "img_a.json").exists()
    assert any("refusing trace append" in r.message for r in caplog.records)


def test_rewrite_yolo_strips_bom(tmp_path) -> None:
    from manual_reviewer.scripts.export_to_aa_v4 import _rewrite_yolo_label
    classes = tmp_path / "classes.txt"
    classes.write_text("﻿cat\ndog\n", encoding="utf-8")
    labels_dir = tmp_path / "labels"

    class _Bbox:
        x1, y1, x2, y2 = 0.1, 0.1, 0.2, 0.2

    class _C:
        class_name = "cat"
        bbox = _Bbox()

    class _Result:
        corrections = [_C()]
        frame_state = "clean"

    _rewrite_yolo_label(labels_dir, "img1", _Result(), classes)
    out = (labels_dir / "img1.txt").read_text(encoding="utf-8")
    assert out.startswith("0 ")  # cat got id 0 even with BOM
