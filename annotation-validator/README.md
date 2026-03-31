# Annotation Validator

Validates YOLO bounding box annotations for class correctness and bbox quality using Qwen3.5 VLM served via vLLM.

Draws a red bbox on the image, sends it to the model in a single pass, and gets back:
- **class_match**: does the object match the claimed label?
- **detected_class**: what the model actually sees
- **bbox_score**: 0-1 bbox quality score (tight coverage, regardless of class)

Results are categorized into **keep / fix / discard** for downstream processing.

## Setup

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env: set HF_TOKEN, NVIDIA_VISIBLE_DEVICES, model choice
```

### 2. Start the vLLM server

```bash
docker compose up -d

# Watch logs until model is loaded (~2-5 min depending on model size)
docker compose logs -f
```

### 3. Install client deps

```bash
pip install -r requirements.txt
```

### 4. Verify server is up

```bash
curl http://localhost:8000/health
```

## Usage

### Single image

```bash
python validator.py --image frame_001.jpg --annotation labels/frame_001.txt
```

### Batch (folder of images + YOLO annotations)

```bash
python validator.py \
    --image-dir /data/images/ \
    --annotation-dir /data/labels/ \
    --output-dir results/
```

### Custom classes

```bash
python validator.py \
    --image-dir images/ \
    --annotation-dir labels/ \
    --output-dir results/ \
    --classes "forklift,pallet_jack" \
    --class-descriptions "A forklift: powered truck with vertical mast and forks. A pallet jack: low-profile wheeled jack, no mast."
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--image` | | Single image path |
| `--annotation` | | Single annotation .txt path (YOLO format) |
| `--image-dir` | | Directory of images for batch mode |
| `--annotation-dir` | | Directory of YOLO .txt labels (defaults to image-dir) |
| `--output-dir` / `-o` | `validation_results` | Output directory |
| `--classes` | `forklift,pallet_jack` | Comma-separated valid class names (order matches YOLO class_id) |
| `--class-descriptions` | built-in | Detailed class descriptions for the model |
| `--thinking` | off | Enable chain-of-thought reasoning (slower) |
| `--workers` | 4 | Concurrent requests to vLLM |

## Input Format

YOLO annotation `.txt` files with one bbox per line:

```
class_id x_center y_center width height
```

All coordinates normalized 0-1. Class IDs map to `--classes` order (0=forklift, 1=pallet_jack by default).

```
0 0.4500 0.5200 0.3000 0.6000
1 0.7200 0.3100 0.1500 0.2500
```

## Output

```
results/
├── sidecar_json/            # detailed JSON per image
│   ├── frame_001.json
│   └── frame_002.json
├── scored_annotations/      # YOLO + bbox_score + category + detected_class
│   ├── frame_001.txt
│   └── frame_002.txt
└── summary.json             # aggregate stats
```

### Sidecar JSON (per image)

```json
{
  "image_path": "/data/images/frame_001.jpg",
  "img_w": 1920, "img_h": 1080,
  "model_id": "Qwen/Qwen3.5-9B",
  "annotations": [
    {
      "bbox_index": 0,
      "class_id": 0,
      "expected_class": "forklift",
      "bbox": [0.45, 0.52, 0.30, 0.60],
      "class_match": false,
      "detected_class": "pallet_jack",
      "bbox_score": 0.85,
      "category": "fix"
    }
  ]
}
```

### Scored annotations (per image)

```
0 0.450000 0.520000 0.300000 0.600000 0.850 fix pallet_jack
```

### Category logic

| class_match | detected_class | bbox_score | Category | Action |
|---|---|---|---|---|
| true | forklift | >= 0.75 | **keep** | Ready for training |
| true | forklift | 0.4 - 0.75 | **fix** | Adjust bbox |
| true | forklift | < 0.4 | **discard** | Unusable bbox |
| false | pallet_jack | >= 0.4 | **fix** | Relabel + maybe adjust bbox |
| false | pallet_jack | < 0.4 | **discard** | Wrong label AND bad bbox |
| false | person | any | **discard** | Not a valid class |

## Model Options

All configured via environment variables (see `.env.example`):

| Model | VRAM | GPUs | Quality | Config |
|---|---|---|---|---|
| **Qwen3.5-9B BF16** | ~18 GB | 1x 24GB | Good for visual classification | `VALIDATOR_MODEL=Qwen/Qwen3.5-9B` `VALIDATOR_TP=1` |
| Qwen3.5-27B FP8 | ~30 GB | 2x 24GB | Higher accuracy | `VALIDATOR_MODEL=Qwen/Qwen3.5-27B-FP8` `VALIDATOR_TP=2` `VALIDATOR_DTYPE=auto` |
| Qwen3.5-27B GPTQ-Int4 | ~16 GB | 1x 24GB | Good, slight quantization loss | `VALIDATOR_MODEL=Qwen/Qwen3.5-27B-GPTQ-Int4` `VALIDATOR_TP=1` `VALIDATOR_DTYPE=auto` |
| Qwen3.5-27B BF16 | ~54 GB | 3-4x 24GB | Best | `VALIDATOR_MODEL=Qwen/Qwen3.5-27B` `VALIDATOR_TP=4` |

### GPU selection

```bash
# Use GPU 0 (default)
NVIDIA_VISIBLE_DEVICES=0 docker compose up -d

# Use GPU 2
NVIDIA_VISIBLE_DEVICES=2 docker compose up -d

# Use GPUs 2 and 3 for 27B FP8
NVIDIA_VISIBLE_DEVICES=2,3 VALIDATOR_MODEL=Qwen/Qwen3.5-27B-FP8 VALIDATOR_TP=2 VALIDATOR_DTYPE=auto docker compose up -d
```

## Environment Variables

### Server (docker-compose)

| Variable | Default | Description |
|---|---|---|
| `HF_HOME` | `/data/all_cache/hf_home` | HuggingFace home directory (shared model cache) |
| `HF_TOKEN` | | HuggingFace auth token |
| `VLLM_PORT` | `8000` | Port for vLLM server |
| `NVIDIA_VISIBLE_DEVICES` | `0` | GPU IDs to expose to the container |
| `VALIDATOR_MODEL` | `Qwen/Qwen3.5-9B` | Model to serve |
| `VALIDATOR_TP` | `1` | Tensor parallel size (number of GPUs) |
| `VALIDATOR_DTYPE` | `bfloat16` | Data type (`bfloat16`, `auto`) |

### Client (validator.py)

| Variable | Default | Description |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server URL |
| `LLM_MODEL` | `Qwen/Qwen3.5-9B` | Model name (must match server) |
| `LLM_API_KEY` | `dummy` | API key (vLLM doesn't need a real one) |
