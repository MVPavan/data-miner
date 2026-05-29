"""Tests for build_tasks CLI helpers: chunked POST + retry, strict skip-existing."""

from __future__ import annotations

from typing import Any

import pytest

from manual_reviewer.scripts import build_tasks as mod


class _FakeResp:
    def __init__(self, status_code: int = 200, text: str = ""):
        self.status_code = status_code
        self.text = text


class _FakeClient:
    def __init__(self, get_resps=None, post_resps=None):
        self._get_resps = list(get_resps or [])
        self._post_resps = list(post_resps or [])
        self.gets: list[dict[str, Any]] = []
        self.posts: list[dict[str, Any]] = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, params=None):
        self.gets.append({"url": url, "params": dict(params or {})})
        return self._get_resps.pop(0) if self._get_resps else _FakeResp(200, "")

    def post(self, url, json=None):
        self.posts.append({"url": url, "json": json})
        return self._post_resps.pop(0) if self._post_resps else _FakeResp(200, "")


def _patch_httpx(monkeypatch, fake):
    class _M:
        Client = lambda *a, **kw: fake  # noqa: E731
    monkeypatch.setitem(__import__("sys").modules, "httpx", _M)


def _tasks(n: int):
    return [{"data": {"image_id": f"img_{i}"}} for i in range(n)]


def test_post_chunks_in_batches(monkeypatch):
    fake = _FakeClient(post_resps=[_FakeResp(200), _FakeResp(200), _FakeResp(200)])
    _patch_httpx(monkeypatch, fake)
    n = mod._post_to_ls(
        _tasks(5),
        base_url="http://x",
        token="t",
        project_id=1,
        skip_existing=False,
        timeout=5.0,
        batch_size=2,
    )
    assert n == 5
    assert len(fake.posts) == 3  # 2 + 2 + 1
    assert [len(p["json"]) for p in fake.posts] == [2, 2, 1]


def test_post_retries_on_5xx(monkeypatch):
    fake = _FakeClient(post_resps=[_FakeResp(503), _FakeResp(503), _FakeResp(200)])
    _patch_httpx(monkeypatch, fake)
    monkeypatch.setattr(mod.time, "sleep", lambda *_a: None)
    n = mod._post_to_ls(
        _tasks(2),
        base_url="http://x",
        token="t",
        project_id=1,
        skip_existing=False,
        timeout=5.0,
        batch_size=10,
    )
    assert n == 2
    assert len(fake.posts) == 3


def test_post_4xx_raises_immediately(monkeypatch):
    fake = _FakeClient(post_resps=[_FakeResp(403, "forbidden")])
    _patch_httpx(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="403"):
        mod._post_to_ls(
            _tasks(1),
            base_url="http://x",
            token="t",
            project_id=1,
            skip_existing=False,
            timeout=5.0,
        )


def test_post_redacts_token_on_4xx(monkeypatch):
    fake = _FakeClient(post_resps=[_FakeResp(401, "bad token=secret_t!")])
    _patch_httpx(monkeypatch, fake)
    with pytest.raises(RuntimeError) as ei:
        mod._post_to_ls(
            _tasks(1),
            base_url="http://x",
            token="secret_t",
            project_id=1,
            skip_existing=False,
            timeout=5.0,
        )
    assert "secret_t" not in str(ei.value)
    assert "<redacted>" in str(ei.value)


def test_strict_skip_existing_raises_on_failure(monkeypatch):
    fake = _FakeClient(get_resps=[_FakeResp(500, "boom")])
    _patch_httpx(monkeypatch, fake)
    import httpx as httpx_mod
    with pytest.raises(RuntimeError, match="LS task list failed 500"):
        mod._fetch_existing_image_ids(
            httpx_mod, "http://x", {"Authorization": "Token t"}, 1, 5.0,
            strict_existing=True,
        )


def test_lenient_skip_existing_falls_back(monkeypatch):
    fake = _FakeClient(get_resps=[_FakeResp(500, "boom")])
    _patch_httpx(monkeypatch, fake)
    import httpx as httpx_mod
    out = mod._fetch_existing_image_ids(
        httpx_mod, "http://x", {"Authorization": "Token t"}, 1, 5.0,
    )
    assert out == set()


def test_argparse_ls_token_from_env(monkeypatch):
    import importlib
    monkeypatch.setenv("LS_TOKEN", "envtok")
    importlib.reload(mod)
    args = mod._parse_args(["--db", "/tmp/x.db", "--ls-url", "http://x", "--ls-project", "1"])
    assert args.ls_token == "envtok"
