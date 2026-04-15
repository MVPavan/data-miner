# aa_v3 — Detector Plugin Refactor (plan)

Goal: make detector models plug-in modules. Add / disable / swap a model by
editing one YAML entry + one Python file. Eliminate every hard-coded model
name string from the pipeline.

---

## Problems we're solving

1. **Hard-coded model names** (`"grounding_dino"`, `"falcon"`, `"sam3"`,
   `"owlvit2"`) scattered across `config.py`, `stages/detect.py`, tests, viewer
   fallbacks.
2. **Per-class looping leaks into the pipeline** (`_call_detector_per_class`
   fires one HTTP POST per `(model, class)`). That's the detector's problem,
   not the pipeline's.
3. **Server and client code live apart**: `servers/serve_*.py` and
   `stages/detect.py::_build_payload_for_class` both know the wire format.
   Drift-prone.
4. **Duplicated server topology**: `configs/default.yaml` AND
   `servers/serve_config.yaml`. Two sources of truth.
5. **Untyped dict shuffling** at every HTTP boundary. No Pydantic validation
   on requests / responses.

---

## Conclusions

### 1. `DetectorName` StrEnum — single source of truth

```python
# contracts.py
class DetectorName(StrEnum):
    GROUNDING_DINO = "grounding_dino"
    FALCON         = "falcon"
    SAM3           = "sam3"
    OWLVIT2        = "owlvit2"
```

Every reference uses `DetectorName.*`. YAML keys must match enum values
(validator); unknown key → load error. `Candidate.source_model` stays a
`str` on the wire, populated from `enum.value`.

### 2. Wire contract — one POST per `(image, model)`, multi-prompt

```
DetectorRequest  = {image_path: str, prompts: list[str]}
DetectorResponse = {boxes, scores, labels}
```

SAM3's `refine` mode is a sibling request shape on the same server, used
only by the refine stage, not by the detection pipeline.

### 3. Adapter = thin client

```python
class BaseHTTPAdapter(DetectorAdapter):
    async def detect(self, session, cfg, image_path, classes):
        req = DetectorRequest(
            image_path=image_path,
            prompts=[c.prompt for c in classes],
        )
        resp = await self._post(session, cfg.port, req)
        return self._to_candidates(resp, classes)
```

- No per-class loop on the client. One HTTP call per `(image, model)`.
- Adapter with a non-standard wire shape overrides `detect()`.
- No `supports_joint_multiclass` flag anywhere.

### 4. Per-class strategy is entirely inside the server

Server's `predict(batch)` decides:
- **Natively multi-prompt** model → one forward pass with the whole
  `prompts` list.
- **Single-prompt** model → internal `for prompt in req.prompts:` loop,
  accumulate boxes, return combined response.

Switching a server from loop to native = override one method on the
subclass. Pipeline never sees the difference.

### 5. Server base class — template method

```python
class DetectorServerBase(ls.LitAPI, ABC):
    request_model:  type[BaseModel] = DetectorRequest
    response_model: type[BaseModel] = DetectorResponse

    def setup(self, device):                     # common: device/dtype
        self.device = device
        self.dtype  = torch.bfloat16 if "cuda" in device else torch.float32
        self._load_model()                       # hook

    def decode_request(self, request):
        req = self.request_model.model_validate(request)
        image = Image.open(req.image_path).convert("RGB")
        return self._prepare(image, req)         # hook

    def predict(self, batch):
        return [self._run_one_request(item) for item in batch]  # overridable

    def encode_response(self, result):
        return self._to_response(result).model_dump()           # hook

    # abstract: _load_model, _prepare, _run_one_request, _to_response
```

Each concrete server becomes ~60 lines vs today's ~200.

### 6. Pydantic everywhere — including tensor-carrying intermediates

No plain `dict`s anywhere. Every I/O shape is a Pydantic model.

- **HTTP boundary** (request / response) — standard JSON-serialisable
  models.
- **Internal stage handoffs** (`decode_request → predict → encode_response`)
  — Pydantic models with `model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)`.
  Validation on these is just an `isinstance()` check (O(1)); overhead is
  unmeasurable relative to a forward pass.
- Internal models **intentionally raise** on accidental JSON dump via a
  `@field_serializer(..., when_used="json")` that rejects tensor fields —
  so crossing a wire boundary with a tensor-carrying model is a loud
  error, not silent data loss.

Rationale: one mental model everywhere, zero loose strings, IDE typing
across the whole pipeline. `pydantic-core`'s Rust validator keeps the
hot-path cost negligible. Research notes in session (April 2026) —
Pydantic v2 stable, v3 tracking issue signals minor-bump only.

### 7. File layout — adapter + server per file

```
stages/detect/
  __init__.py
  adapters/
    __init__.py           # ADAPTERS: dict[DetectorName, DetectorAdapter]
    base.py               # DetectorAdapter ABC + BaseHTTPAdapter
    grounding_dino.py     # GDINOAdapter + GDINOServer
    falcon.py             # FalconAdapter + FalconServer
    sam3.py               # SAM3Adapter + SAM3Server  (+ refine-mode code)
    owlvit2.py            # OWLv2Adapter + OWLv2Server
servers/
  base.py                 # DetectorServerBase(ls.LitAPI)
  launch_all.py           # reads configs/default.yaml::servers.detectors
  # serve_*.py            # DELETED — content moved to adapters/*.py
  # serve_config.yaml     # DELETED — single config in default.yaml
```

Adding a new detector = one new file in `stages/detect/adapters/` +
YAML entry. Nothing else touched.

### 8. Config shape — unified, single source

```yaml
servers:
  vlm: { … }
  detectors:
    grounding_dino:       # key validated against DetectorName
      enabled: true
      port: 3001
      gpu: cuda:0
      model_id: IDEA-Research/grounding-dino-base
      script: serve_gdino.py
      max_batch_size: 8
      batch_timeout_ms: 50
    falcon:
      enabled: true
      port: 3002
      gpu: cuda:0
      …
    sam3:
      enabled: true
      port: 3003
      gpu: cuda:1
      …
    owlvit2:
      enabled: false      # parked but discoverable
      …
```

Enable/disable = one-line YAML change. `launch_all.py` reads the same
block, filters by `enabled`.

### 9. Batching — unchanged

Three layers, status:

| Layer | State |
|---|---|
| Client concurrency — `asyncio.gather` over `(image × model)` | Active |
| LitServe request coalescing — `max_batch_size` / `batch_timeout_ms` | Active (amortises HTTP/Python) |
| Per-request inference — sequential loop over batch items and inner prompts | Active (no tensor-stacked batching) |
| True tensor-stacked batching | Deferred; per-adapter opt-in later |

Client fires one POST per `(image, model)` with the full class list; N
images run concurrently. Server loops internally. Throughput today is
gated by per-image inference time, not our batching.

---

## Where hard-coded strings go

| Before | After |
|---|---|
| `if model_name == "grounding_dino":` | `ADAPTERS[DetectorName.GROUNDING_DINO]` lookup — one place |
| `for name in ("grounding_dino","falcon","sam3"):` | `for name, cfg in config.servers.detectors.items() if cfg.enabled` |
| `self.config.servers.sam3.port` (refine.py) | `self.config.servers.detectors[DetectorName.SAM3].port` (refine is SAM-specific by design) |
| `source_model: "sam3"` | `source_model=DetectorName.SAM3.value` (string on wire, enum in code) |

---

## Files — change list

- **`contracts.py`** — add `DetectorName`, `DetectorRequest`,
  `DetectorResponse`, `SAM3RefineRequest/Response`.
- **`config.py`** — `ServersConfig.detectors: dict[DetectorName,
  DetectorConfig]` with key validator; drop typed `grounding_dino/…`
  fields; add `enabled`, `script` to `DetectorConfig`.
- **`stages/detect.py`** — slim down: call `ADAPTERS[name].detect(...)`
  per enabled detector; delete `_build_payload_for_class`,
  `_call_detector_per_class`, `_call_one_class`,
  `_parse_response_for_class`.
- **`stages/detect/adapters/*.py`** — new: one file per model, each
  containing an `Adapter` (thin HTTP client) and a `Server` (LitServe
  subclass of `DetectorServerBase`).
- **`servers/base.py`** — new: `DetectorServerBase`.
- **`servers/serve_*.py`** — delete.
- **`servers/serve_config.yaml`** — delete.
- **`servers/launch_all.py`** — rewrite to load
  `configs/default.yaml → servers.detectors`, filter by `enabled`.
- **`stages/refine.py`** — replace `servers.sam3.port` with
  `servers.detectors[DetectorName.SAM3].port`.
- **`configs/default.yaml`** — restructure `servers:` as above;
  `owlvit2.enabled: false`.
- **viewer** — no change (MODEL_COLORS fallback keeps working).
- **tests** — `test_multiclass_proposal.py`, `compare_litserve.py`:
  ports + names come from config; remove joint/per-class divergence
  (servers now handle that internally).

---

## Deferred / out of scope

- True tensor-stacked batching. Per-adapter opt-in by overriding
  `DetectorServerBase.predict(batch)`; no architectural block.
- SAM3 native multi-prompt. Model may grow this — if so, the SAM3
  server subclass implements `_run_one_request` to batch prompts.
  No pipeline change needed.
- Evaluate-stage VLM failures (separate thread, tracked in the
  filtering/scoring doc under "per-candidate classify" — still being
  worked).
