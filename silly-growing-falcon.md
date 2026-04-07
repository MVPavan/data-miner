# Agentic Auto-Annotation Pipeline: VLM + Falcon Perception + SAM3

## Context

**Goal**: Build an agentic auto-annotation system where Qwen 3.5 VL orchestrates Falcon Perception and SAM 3.1 in a reasoning loop to achieve best-possible annotation accuracy for any class.

**Current state**: The data_miner pipeline runs detectors independently (GroundingDINO, Falcon Perception, SAM3, OWLv2, etc.), produces YOLO annotations, then post-hoc validates with Qwen 3.5 VLM (`annotation-validator/validator.py`). No reasoning loop between detection and validation.

**Inspiration**: `Falcon-Perception-upstream/demo/perception_agent.ipynb` shows an agentic loop where a VLM orchestrator reasons, calls Falcon Perception tools, verifies results, and refines iteratively.

**User requirements**:
- Support **two modes**: hybrid consensus + agentic AND full agentic loop (configurable)
- Use **Qwen 3.5 VL** as VLM orchestrator (via existing vLLM setup)
- Configurable detector set (Falcon, SAM3, GroundingDINO, OWLv2 -- any subset)

---

## Research-Informed Critical Analysis

### What the research says (2024-2026)

| System | Type | Performance | Key Lesson |
|--------|------|-------------|------------|
| SAM 3 Agent (Meta) | Prompt loop | ReasonSeg 76.0 gIoU (SOTA) | SAM3's vocabulary matters more than agent sophistication |
| RSAgent (arXiv 2512.24023) | RL-trained MDP, 8 turns | ReasonSeg 66.5 gIoU | Process rewards: penalize unnecessary tool calls |
| Real-LOD (ICLR 2025) | 5-state machine, 4 cycles | OmniLabel +50% AP | State machine > flat retry loop |
| IR-SIS (arXiv 2602.09252) | SAM3 + Qwen2.5-VL-32B evaluator | EndoVis +7.8 IoU OOD | Quantitative quality gates > "VLM, does this look right?" |
| Grounded SAM 2 | Deterministic pipeline | COCO 54.3 AP, SegInW 48.7 AP | **Still the gold standard for standard annotation** |
| ComfyUI-Seg-Agent | VLM + SAM3 retry loop | No benchmarks | Author: "often worse and much slower than Grounded SAM" |
| PyImageSearch tutorial | Qwen2.5-VL + SAM3, 3 rounds | No benchmarks | Took all 3 rounds to segment "a bag on the leftmost side" |

### Uncomfortable truths

1. **No agentic system has beaten Grounded SAM 2 for standard auto-annotation.** Agentic systems shine on *reasoning* tasks (complex NL queries), not bread-and-butter detection.
2. **Simple VLM retry loops can be worse than deterministic pipelines.** The ComfyUI author says this explicitly. VLM hallucination can be amplified by retry loops.
3. **SAM3's concept vocabulary matters more than agent sophistication.** SAM3 Agent (simple loop) gets 76.0 gIoU; RL-trained RSAgent gets 66.5.
4. **The real win is model consensus, not agentic reasoning** (arXiv:2504.11101): Inter-model agreement entropy improved F1 by 42% over VLM-as-Judge baselines.

### Design principles (derived from research)

1. **Quantitative quality gates** (from IR-SIS): Use hard IoU/coverage thresholds as termination conditions, not soft VLM judgments
2. **State machine > retry loop** (from Real-LOD): Plan -> Detect -> Reflect -> Decide
3. **Penalize unnecessary iterations** (from RSAgent): Bias toward accepting good results early
4. **Consensus first, VLM second** (from consensus entropy paper): Multi-model agreement is cheaper and more reliable than VLM verification

---

## Architecture: Two Modes

### Mode A: Hybrid Consensus + Targeted Agentic Refinement (default, production)

```
Phase 1: Multi-Model Detection (parallel)
┌──────────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────┐
│    Falcon     │  │   SAM3   │  │ GroundingDINO│  │  OWLv2   │
│  Perception   │  │          │  │              │  │ (optional)│
└──────┬───────┘  └────┬─────┘  └──────┬───────┘  └────┬─────┘
       └───────────────┬────────────────┘───────────────┘
                       ▼
Phase 2: Consensus Merge + Quality Gates
┌─────────────────────────────────────────────────────────────┐
│  Match detections across models (IoU > 0.5)                  │
│  Quality gate: coverage + overlap + bbox consistency         │
│  → Agreed (2+ models) + quality pass: Auto-accept            │
│  → Single-model with high confidence: Auto-accept            │
│  → Below quality gate: Flag for VLM review                   │
│  → Conflicting labels: Flag for VLM tiebreak                 │
└────────────────────────┬────────────────────────────────────┘
                         ▼
Phase 3: Agentic VLM Refinement (only flagged ~10-20%)
┌─────────────────────────────────────────────────────────────┐
│  Qwen 3.5 VL Orchestrator (via vLLM)                        │
│                                                              │
│  State machine per flagged detection (max 3 iterations):     │
│  VERIFY -> ACCEPT | REJECT | RE_SEGMENT                     │
│    1. Show image crop + candidate bbox overlay               │
│    2. VLM decides with structured JSON output                │
│    3. If RE_SEGMENT: call detector with VLM-designed prompt  │
│    4. Quality gate check on new result                       │
│    5. Auto-accept if passes gate; else iterate               │
└────────────────────────┬────────────────────────────────────┘
                         ▼
Phase 4: Merge & Output (YOLO labels + provenance JSON)
```

### Mode B: Full Agentic Loop (every image, experimental)

**WARNING**: Research shows this is slower and often no better than Mode A for standard annotation. Use only for: small datasets (<100 images), novel/ambiguous classes, or complex compositional queries.

```
For each image (state machine, max 8 tool calls):
┌─────────────────────────────────────────────────────────────┐
│  Qwen 3.5 VL Orchestrator                                   │
│                                                              │
│  States: PLAN -> DETECT -> REFLECT -> DECIDE                 │
│                                                              │
│  PLAN: VLM analyzes image, decides detection strategy        │
│  DETECT: VLM calls detect_objects(class, model) tool         │
│  REFLECT: VLM receives detections, applies quality gates     │
│  DECIDE: accept / reject / re-segment / next class           │
│                                                              │
│  Quality gates (quantitative, not VLM opinion):              │
│  - bbox area: 0.001 < area_fraction < 0.9                   │
│  - confidence: > threshold                                   │
│  - edge proximity: not touching 2+ edges                     │
│  - duplicate check: IoU < 0.8 with existing detections       │
│                                                              │
│  Tools: detect_objects, verify_detection, re_segment,        │
│         get_crop, compute_relations, finalize                │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Step 1: Consensus Detection Engine

**New file**: `data_miner/models/consensus_detector.py`

```python
class ConsensusDetector:
    """Multi-model consensus detection with configurable detector set."""
    
    def __init__(self, detectors: list[str], detection_classes: list[str],
                 iou_threshold=0.5, min_agreement=2):
        # detectors: subset of ["falcon", "sam3", "grounding_dino", "owlv2"]
        # Lazy-loads only requested models
    
    def detect_single(self, image) -> ConsensusResult:
        # 1. Run each detector on image
        # 2. Normalize all outputs to common format (xyxy normalized)
        # 3. Match detections across models (Hungarian matching on IoU)
        # 4. Apply quality gates to each detection
        # 5. Categorize: agreed / single_model / label_conflict
        # 6. Weighted box fusion for agreed detections
    
    def detect_batch(self, images, batch_size=4) -> list[ConsensusResult]:
        # Batch processing with progress bar
```

**Key data structures**:
```python
@dataclass
class ConsensusDetection:
    bbox: tuple[float, float, float, float]  # normalized xyxy
    label: str
    confidence: float                         # average across agreeing models
    sources: list[str]                        # ["falcon", "sam3"]
    status: str                               # "agreed" | "single_model" | "label_conflict"
    quality_score: float                      # from quality gate (0-1)
    source_detections: dict                   # raw per-model detections

@dataclass  
class ConsensusResult:
    accepted: list[ConsensusDetection]        # passed consensus + quality gate
    flagged: list[ConsensusDetection]         # needs VLM review
    rejected: list[ConsensusDetection]        # failed quality gate hard
    image_path: Path
    stats: dict                               # agreement rate, per-model counts, etc.
```

**Quality gates** (quantitative, from IR-SIS pattern):
```python
def quality_gate(detection) -> tuple[bool, float]:
    """Hard metric thresholds, not VLM opinion."""
    x1, y1, x2, y2 = detection.bbox
    w, h = x2 - x1, y2 - y1
    area = w * h
    
    checks = {
        "min_area": area > 0.001,
        "max_area": area < 0.9,
        "min_dimension": min(w, h) > 0.005,
        "not_edge_hugging": not touches_multiple_edges(detection.bbox),
        "aspect_ratio": 0.05 < (w / max(h, 1e-6)) < 20.0,
    }
    score = sum(checks.values()) / len(checks)
    passes = all(checks.values())
    return passes, score
```

**Matching algorithm**: 
- Compute IoU matrix between all detections from all models
- Hungarian matching (scipy.optimize.linear_sum_assignment) with IoU > threshold
- Group matched detections by overlap cluster
- If N sources >= min_agreement -> accepted
- Weighted box fusion: average bbox coords weighted by model confidence

Reuses: `detection_utils.py:apply_nms`, existing detector helpers

### Step 2: Agentic VLM Annotator

**New file**: `data_miner/models/agentic_annotator.py`

```python
class AgenticAnnotator:
    """VLM-orchestrated annotation refinement using Qwen 3.5 VL."""
    
    def __init__(self, vlm_config, detector_helpers: dict,
                 max_iterations=3, mode="hybrid"):
        # vlm_config: vLLM connection (reuse annotation-validator pattern)
        # detector_helpers: {"falcon": FalconHelper, "sam3": SAMHelper, ...}
    
    def refine_flagged(self, image, flagged_detections) -> list[FinalDetection]:
        """Mode A: Refine only flagged detections from consensus.
        
        State machine per detection: VERIFY -> ACCEPT | REJECT | RE_SEGMENT
        Quality gate applied after re-segmentation (auto-accept if passes).
        """
    
    def annotate_full(self, image, detection_classes) -> list[FinalDetection]:
        """Mode B: Full agentic loop for one image.
        
        State machine: PLAN -> DETECT -> REFLECT -> DECIDE
        Penalizes unnecessary iterations (bias toward early acceptance).
        """
```

**VLM connection** (reuses `annotation-validator/validator.py` pattern):
```python
# Same OpenAI SDK + vLLM setup already in docker-compose-vllm.yml
client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

# Multi-modal message format (same as validator.py lines 405-424)
response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": prompt}
    ]}],
    max_tokens=512,
    temperature=0.7,
    extra_body={"enable_thinking": True}
)
```

**Tool definitions for Mode B** (adapted from perception_agent + Real-LOD state machine):

| Tool | Parameters | Returns |
|------|-----------|---------|
| `detect_objects` | `{"class": str, "model": "falcon"\|"sam3"\|"auto"}` | Detection list + SoM overlay |
| `verify_detection` | `{"detection_id": int}` | Cropped image + quality metrics |
| `re_segment` | `{"expression": str, "model": "falcon"\|"sam3"}` | New detection list |
| `compute_relations` | `{"detection_ids": [int, int]}` | Pairwise IoU, positions |
| `finalize` | `{"accepted": [int], "rejected": [int]}` | Final annotation set |

**System prompts** (two variants):

Mode A (refinement) -- concise, single-turn per detection:
```
You are an annotation quality specialist verifying automated detections.

You see an image crop with a RED bounding box and metadata:
- Which models produced this detection
- Quality gate score
- Expected class label

Decide ONE of:
- accept: bbox correctly contains the labeled object
- reject: false positive, wrong object, or quality too poor
- re_segment: try detecting with expression "{expression}" using {model}

JSON only: {"action": "accept|reject|re_segment", "class": "...",
            "expression": "...", "model": "falcon|sam3", "reason": "..."}
```

Mode B (full agentic) -- multi-turn, Real-LOD state machine pattern:
```
You are an auto-annotation agent. Find and annotate all instances 
of the target classes in this image.

Target classes: {class_list}

STRATEGY (follow this order):
1. Call detect_objects for each target class (start broad)
2. Review detections: quality_score > 0.8 can be auto-accepted
3. For quality_score < 0.8: call verify_detection to see crop
4. If detection looks wrong: call re_segment with simpler expression
5. Use compute_relations to check duplicates (IoU > 0.5)
6. Call finalize when all classes processed

RULES:
- Call exactly ONE tool per turn
- Bias toward accepting: if quality_score > 0.8, accept without verifying
- Max 8 tool calls total. Don't over-iterate.

<think>
Reasoning about what to do next.
</think>
<tool>{"name": "...", "parameters": {...}}</tool>
```

**JSON parsing**: Reuse `validator.py:parse_json_output()` -- handles thinking blocks, markdown fences, malformed JSON.

### Step 3: Integration Script

**New file**: `scripts/run_agentic_annotate.py`

```python
"""
Run agentic auto-annotation pipeline.

Usage:
    # Mode A: Consensus + targeted VLM refinement (recommended)
    python scripts/run_agentic_annotate.py --mode hybrid \
        --detectors falcon sam3 grounding_dino \
        --classes forklift "pallet jack" \
        --input-dir /path/to/images --output-dir /path/to/output

    # Mode B: Full agentic loop (experimental, slow)
    python scripts/run_agentic_annotate.py --mode full \
        --detectors falcon sam3 \
        --classes forklift "pallet jack" \
        --input-dir /path/to/images --output-dir /path/to/output

    # Consensus only, no VLM (fastest)
    python scripts/run_agentic_annotate.py --mode consensus \
        --detectors falcon sam3 grounding_dino \
        --classes forklift "pallet jack"

    # Sanity check (5 images)
    python scripts/run_agentic_annotate.py --mode hybrid --sanity
"""
```

Pipeline flow:
1. Parse args, load config
2. Initialize detector helpers (only requested models)
3. If mode="consensus":
   a. Run `ConsensusDetector.detect_batch()` on all images
   b. Save accepted consensus detections as YOLO labels
   c. Report flagged items (no VLM)
4. If mode="hybrid":
   a. Run `ConsensusDetector.detect_batch()` on all images
   b. Auto-save accepted consensus detections
   c. Run `AgenticAnnotator.refine_flagged()` on flagged items
   d. Merge refined with accepted
5. If mode="full":
   a. For each image: `AgenticAnnotator.annotate_full()`
6. Save YOLO labels + provenance JSON per image
7. Save summary statistics (agreement rate, VLM call count, etc.)

### Step 4: Config & Constants

**Modify** `data_miner/config/constants.py`:
- Add `AnnotationMode` enum: `CONSENSUS`, `HYBRID`, `FULL_AGENTIC`

**Modify** `data_miner/config/config.py`:
```python
class ConsensusConfig(BaseModel):
    detectors: list[str] = ["falcon", "sam3", "grounding_dino"]
    iou_threshold: float = 0.5
    min_agreement: int = 2
    quality_gate_min_area: float = 0.001
    quality_gate_max_area: float = 0.9

class AgenticConfig(BaseModel):
    mode: str = "hybrid"  # "consensus", "hybrid", or "full"
    vlm_base_url: str = "http://localhost:8005/v1"
    vlm_model: str = "Qwen/Qwen3.5-27B-FP8"
    max_iterations: int = 3       # per flagged detection (hybrid)
    max_tool_calls: int = 8       # per image (full mode)
    enable_thinking: bool = True
    detection_threshold: float = 0.3
    auto_accept_quality: float = 0.8  # quality score above this = no VLM needed
```

**Modify** `data_miner/config/default.yaml`:
```yaml
agentic_annotate:
  mode: "hybrid"
  consensus:
    detectors: ["falcon", "sam3", "grounding_dino"]
    iou_threshold: 0.5
    min_agreement: 2
  vlm:
    base_url: "http://localhost:8005/v1"
    model: "Qwen/Qwen3.5-27B-FP8"
    max_iterations: 3
    max_tool_calls: 8
    enable_thinking: true
    auto_accept_quality: 0.8
```

---

## Key Files

| File | Action | Purpose |
|------|--------|---------|
| `data_miner/models/consensus_detector.py` | **Create** | Multi-model consensus matching + quality gates |
| `data_miner/models/agentic_annotator.py` | **Create** | VLM reasoning loop (both modes) |
| `scripts/run_agentic_annotate.py` | **Create** | CLI entry point with 3 modes |
| `data_miner/config/config.py` | **Modify** | Add ConsensusConfig, AgenticConfig |
| `data_miner/config/constants.py` | **Modify** | Add AnnotationMode enum |
| `data_miner/config/default.yaml` | **Modify** | Add agentic_annotate section |
| `data_miner/models/__init__.py` | **Modify** | Export new classes |

**Reused code** (not modified):
- `falcon_perception.py:FalconPerceptionHelper` -- detect/detect_batch
- `sam.py:SAMHelper` -- detect/segment_with_text
- `grounding_dino.py:GroundingDINOHelper` -- detect
- `owlvit.py:OWLViTHelper` -- detect (optional)
- `detection_utils.py` -- NMS, format_detections, visualization
- `annotation-validator/validator.py` -- vLLM connection pattern, parse_json_output, image_to_base64, draw_bbox

---

## Verification Plan

1. **Sanity test** (5 images): `python scripts/run_agentic_annotate.py --sanity --mode hybrid`
   - Verify consensus matching produces sensible results
   - Verify VLM is called only for flagged detections
   - Check YOLO output format matches existing pipeline

2. **Mode comparison** (50 images):
   - Run: single Falcon, single GroundingDINO, consensus-only, hybrid, full agentic
   - Compare detection count, precision (manual spot-check), latency

3. **Against existing validator** (100 images):
   - Run through current pipeline (detect + validator.py)
   - Run through new agentic pipeline
   - Compare keep/fix/discard ratios

4. **Full dataset**: Run on forklift/pallet_jack dataset, spot-check results

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Falcon hallucination (MCC 0.64) | Consensus requires 2+ model agreement; quality gates filter bad detections |
| VLM retry loop amplifies errors | Quality gates (quantitative) override VLM opinion; max 3 iterations |
| VLM latency on large datasets | Hybrid: VLM only on ~15% of images; consensus mode skips VLM entirely |
| vLLM + detectors compete for GPU | Sequential: run all detectors first, then VLM batch |
| Mode B slower than deterministic pipeline | Clearly marked experimental; default is Mode A |
| Prompt sensitivity per class | Configurable prompt templates; VLM can design alternative prompts |
