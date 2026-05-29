# Legacy Prompt Archive

This directory preserves useful prompt text from older or adjacent annotation
tools before legacy implementation code is retired. It is prompt-only on
purpose: no model code, checkpoint logic, outputs, logs, or generated data
belongs here.

## Rules

- `data_miner/auto_annotation_v4/` remains the active authority.
- Keep provenance in every archived prompt file: source path, old version or
  tool, and the original callable/config section when applicable.
- Do not copy runtime artifacts or gitignored data into this archive.
- If a legacy prompt is byte-for-byte identical to an active v4 prompt, record
  that in this README instead of duplicating the file.

## Current Audit

- `data_miner/auto_annotation_v3/prompts/v1/*.yaml` was compared against
  `data_miner/auto_annotation_v4/prompts/v1/*.yaml`; all prompt YAML files
  matched byte-for-byte at archive creation time. They are already preserved
  by the active v4 prompt set.
- `data_miner/auto_annotation/prompts.py` contributed
  `v0_verification_prompt.yaml`.
- `data_miner/auto_annotation_v2/agents/reasoning.py` contributed
  `v2_reasoning_prompts.yaml`.
- `data_miner/auto_annotation_v2/fl_pj.yaml` contributed
  `v2_forklift_palletjack_prompt_variants.yaml`.
- `annotation-validator/validator.py` contributed
  `annotation_validator_forklift_palletjack.yaml` because its forklift vs.
  pallet-jack class descriptions are a useful prompt asset.