# Legacy Auto-Annotation Cleanup Audit

Status: completed 2026-05-14.

This audit records removal of older auto-annotation implementations. It covers
tracked source and docs only. Gitignored runtime data, outputs, logs, exports,
scratchpad work, run configs, and local datasets were protected and were not
cleaned, moved, archived, or deleted.

## Source Of Truth

`data_miner/auto_annotation_v4/` is the active authority for auto-annotation
pipeline code, checkpoint DB contracts, model-server integration, and active
prompts.

Reusable prompt text from older systems is preserved under
`data_miner/auto_annotation_v4/prompts/archive/legacy/`.

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

Decision: removed after explicit user approval to delete v1/v2/v3 source code.

Former external tracked references, removed with this cleanup:

- `scripts/compare_proposals.py` imports v2 config and proposal helpers for
  model-output comparison.
- `data_miner/auto_annotation_v3/tests/compare_litserve.py` imports v2 config
  and proposal helpers for LitServe parity comparisons.

Prompt preservation: v2 reasoning prompts and forklift/pallet-jack prompt
variants are already copied into the v4 legacy prompt archive.

Removal result: the tracked package directory was deleted. `scripts/compare_proposals.py`
was deleted because it compared standalone model helpers against v2 proposal
internals.

### `data_miner/auto_annotation_v3/`

Decision: removed after explicit user approval to delete v1/v2/v3 source code.

Former tracked references included parity tests, benchmark scripts, migration
plans, and stale verification guidance. Those tracked source paths and stale
docs were deleted or rewritten to point at v4.

Prompt preservation: v3 prompt YAML files under `prompts/v1/` were compared
against v4 `prompts/v1/` and were byte-for-byte identical at audit time, so
the active v4 prompt set already preserves them.

Removal result: the tracked package directory was deleted. Stale v3 migration
plans and v3 tuning docs were also removed.

## Remaining Policy

`data_miner/auto_annotation_v4/` is now the only retained auto-annotation
engine source package. Prompt provenance references under the v4 legacy prompt
archive are intentionally retained.