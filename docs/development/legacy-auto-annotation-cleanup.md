# Legacy Auto-Annotation Cleanup Audit

Status: in progress 2026-05-13.

This audit gates removal of older auto-annotation implementations. It covers
tracked source and docs only. Gitignored runtime data, outputs, logs, exports,
scratchpad work, run configs, and local datasets are protected and must not be
cleaned, moved, archived, or deleted without explicit user review.

## Source Of Truth

`data_miner/auto_annotation_v4/` is the active authority for auto-annotation
pipeline code, checkpoint DB contracts, model-server integration, and active
prompts.

Reusable prompt text from older systems is preserved under
`data_miner/auto_annotation_v4/prompts/archive/legacy/` before implementation
cleanup proceeds.

## Audit Commands

The current audit used tracked-file searches only:

```bash
git grep -n -E 'data_miner\.auto_annotation($|[^_])|data_miner/auto_annotation(/|\b)' \
  -- ':!data_miner/auto_annotation/**' \
     ':!data_miner/auto_annotation_v4/prompts/archive/legacy/**' \
     '*.py' '*.md' '*.yaml' '*.yml' '*.toml' '*.sh'

git grep -n 'auto_annotation_v2' \
  -- ':!data_miner/auto_annotation_v2/**' \
     ':!data_miner/auto_annotation_v4/prompts/archive/legacy/**' \
     '*.py' '*.md' '*.yaml' '*.yml' '*.toml' '*.sh'

git grep -n 'auto_annotation_v3' \
  -- ':!data_miner/auto_annotation_v3/**' \
     ':!data_miner/auto_annotation_v4/prompts/archive/legacy/**' \
     '*.py' '*.md' '*.yaml' '*.yml' '*.toml' '*.sh'
```

## Package Decisions

### `data_miner/auto_annotation/`

Decision: removed after prompt preservation and external-reference audit.

External tracked references: none found outside the package itself, excluding
the prompt archive provenance notes.

Prompt preservation: `data_miner/auto_annotation/prompts.py` contributed
`v0_verification_prompt.yaml` to the v4 legacy prompt archive.

Removal result: the tracked package directory was deleted. No ignored outputs,
logs, exports, scratchpad files, run directories, or datasets were touched.

### `data_miner/auto_annotation_v2/`

Decision: blocked for now.

External tracked references:

- `scripts/compare_proposals.py` imports v2 config and proposal helpers for
  model-output comparison.
- `data_miner/auto_annotation_v3/tests/compare_litserve.py` imports v2 config
  and proposal helpers for LitServe parity comparisons.

Prompt preservation: v2 reasoning prompts and forklift/pallet-jack prompt
variants are already copied into the v4 legacy prompt archive.

Removal rule: migrate or retire the comparison scripts before deleting v2.

### `data_miner/auto_annotation_v3/`

Decision: blocked for now.

External tracked references show v3 is still documented as the legacy parity
engine and still appears in verification guidance, migration plans, v4
replacement comments, and v4 model provenance comments.

Prompt preservation: v3 prompt YAML files under `prompts/v1/` were compared
against v4 `prompts/v1/` and were byte-for-byte identical at audit time, so
the active v4 prompt set already preserves them.

Removal rule: remove or rewrite v3 parity tests, docs, and verification entries
only after the project no longer needs v3 as a benchmark/reference.

## Next Safe Step

Do not remove v2 or v3 yet. The next cleanup phase should either:

1. Migrate or retire `scripts/compare_proposals.py` and the v3
  `compare_litserve.py` dependencies on v2, then re-audit v2; or
2. Leave v2/v3 in place and move to non-destructive consolidation work such as
  shared annotation exchange models for LS/CVAT.