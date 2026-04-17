Client / Per-Request Settings
ParameterFromToWhytemperature0.00.1Qwen explicitly warns against greedy decoding (repetition loops)max_tokens (client default)4096256Actual output ~100-200 tokens; 4096 over-reserves scheduler slotsenable_thinkingfalsekeep falseClassification task, not reasoning — thinking adds 20-40% token overhead with no benefit
Prompt Template max_tokens Cleanup
TemplateCurrentRecommendedclassify_one.yaml384256classify_one_with_crop.yaml384256refine_prompt.yaml384384 (keep)refine_adjudicate.yaml256256 (keep)
Also: the grouped templates (classify_industrial, classify_luggage, etc.) in prompts/v1/ are dead code — current flow only calls classify_one / classify_one_with_crop. Clean up when convenient.
Token Budget Analysis
Per call type, given your image encoding path (pil_to_data_url(max_size=1024), Qwen3-VL 32×32 patches):
CallImagesInputOutputTotal p99classify_one1 overview~1,350200~1,750classify_one_with_cropoverview + crop~2,000200~2,200 ← worst caserefine_prompt1 crop~800384~1,200refine_adjudicate1 crop~800256~1,050
max_model_len: 3072 = worst case 2,200 + 400 edge case + 472 safety margin.
Image Encoding Fix (pil_to_data_url)
Issues with current implementation:

Saves as PNG (lossless) → 10× larger payloads than JPEG q=90 for zero VLM quality benefit
Same max_size=1024 for both overview and crop → wastes tokens on already-zoomed crops

Fix: Switch to JPEG quality=90, use max_size=1024 for overview and max_size=512 for crops. ~10× smaller HTTP payloads, ~20-30% fewer image tokens on crop-using calls, no measurable accuracy loss.
Pipeline-Side Changes
Bump evaluate stage semaphore to match new capacity:
yamlevaluate:
  concurrency: 64    # was ~8-16, can push to 64-128 with 4× DP capacity
Validation Before Deployment

Token distribution check: grep vlm_total_tokens from existing checkpoints to confirm p99 < 2,500. If higher, raise max_model_len to 4096.
Temperature change risk: grep evaluate logs for "Malformed verdict" rate — if >1%, the temperature=0.0 greedy repetition issue was already biting you silently.
JPEG A/B test: run ~50 candidates through current (PNG) vs new (JPEG q=90) encoders, confirm >98% verdict agreement. Bump to q=95 if any drift.
vLLM version: pin to a specific tag with confirmed Qwen3.5 support (check release notes) rather than latest.