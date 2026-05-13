# Adoption Report

Status: adopted for v2.7 on 2026-04-07

## Inputs Read

- `project_agnostic_claude_setup/AGENTS.md`
- `project_agnostic_claude_setup/adopt.md`
- `project_agnostic_claude_setup/.claude/project/*.md`
- `docs/design/v_2_7/core/bodha-design-v2_7.md`
- `docs/design/v_2_7/core/bodha-buddhi-v2_7.md`
- `docs/design/v_2_7/core/bodha-chitta-v2_7.md`
- `docs/design/v_2_7/core/bodha-dharana-v2_7.md`
- `docs/design/v_2_7/core/bodha-dhriti-v2_7.md`
- `docs/design/v_2_7/core/bodha-infrastructure-v2_7.md`
- `docs/design/v_2_7/core/bodha-job-plan-v2_7.md`
- `docs/design/v_2_7/core/bodha-rag-v2_7.md`
- `docs/design/v_2_7/core/bodha-retrieval-v2_7.md`
- `docs/design/v_2_7/core/bodha-tools-skills-v2_7.md`
- `docs/design/v_2_7/core/memory-benchmarks-v2_7.md`
- `docs/design/v_2_6/core/new_discussions/claude-mem-adoption-log.md`
- `docs/design/v_2_6/core/new_discussions/dhriti-extraction-discussions.md`
- `docs/design/v_2_6/core/bodha_subscription_proxy_architecture.md`
- `bodha_claude_setup_old/.claude/project/*.md` (reference for prior adoption)

## Files Updated

- `.claude/project/brief.md`
- `.claude/project/components.md`
- `.claude/project/verification.md`
- `.claude/project/invariants.md`
- `.claude/project/docs-index.md`
- `.claude/project/learnings.md`
- `.claude/project/adoption-report.md`

## Assumptions

- The eventual Bodha implementation remains Python-first and async, based on the authoritative v2.7 design docs.
- All paths in this overlay are repo-relative to the Bodha project root.
- The v2.7 design set supersedes the older v2.2-era harness whenever they conflict.

## Conflicts or Gaps

- Current repo reality is docs-first: no source tree, manifests, CI config, or test directories were found. Infrastructure setup exists under `infra/proxy-setup/`.
- The old setup encodes stale v2.2 assumptions that conflict with v2.7, including `async_core.Scheduler`, OmegaConf-based config guidance, and the older Dhī boundary.
- The v2.7 design docs live under `docs/design/v_2_7/core/`, while supporting discussion docs live under `docs/design/v_2_6/core/new_discussions/`.
- Because repo-native verification commands do not exist yet, the adopted verification surface is limited to doc-consistency checks plus explicit future Python fallback.

## Next Review Step

- Human review required before treating the setup as fully adopted.
- When Bodha source lands, replace doc-only verification with repo-native commands and promote code-level invariants from design intent into mechanically checkable checks.
