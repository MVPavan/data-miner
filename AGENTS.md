
# Coding guideline
1. Follow `.claude/skills/ak-guide/SKILL.MD` for coding guidelines that reduce common LLM mistakes.

2. Prefer html-artifacts for human-facing artifacts, refer to `.claude/skills/html-artifact/` for better understanding.


# Agent Operating Guide

Core harness is stable. Repo-specific facts live in `.claude/project/`.
Codex will review your output once you are done at any task.

## Read Order

1. `AGENTS.md`
2. `.claude/project/brief.md`, `docs-index.md`, `verification.md`, `invariants.md`
3. `docs/roadmap.md`, `docs/status.md` (when doing implementation work)
4. Relevant rules under `.claude/rules/`

## Working Mode

Classify the task before acting.

- `small`: 1-2 files, low ambiguity, reversible. Execute directly, then self-check.
- `standard`: bounded feature, bug fix, or refactor. Short plan before coding.
- `deep`: cross-cutting, high-risk, or ambiguous. Brainstorm, plan, review, execute via subagents, capture learnings.

Lean by default. Match ceremony to scope and risk.

## Process Before Execution

- unclear or exploratory request: brainstorm first
- approved requirements plus multi-step code work: plan first
- newly written requirements or plan docs: review the document before execution
- risky behavior change or fragile legacy area: test-first or characterization-first
- bug, failure, or confusing behavior: systematic-debugging before proposing fixes
- approved plan with bounded tasks: subagent-driven development
- about to claim success: verify before completion

If the user already supplied a clear, approved plan, do not re-run brainstorming.

## Phase Execution

For implementation work, use `/phase-execution N` to drive the full cycle. It delegates to planning, subagent-driven-development, TDD, debugging, and verification skills automatically. See `docs/roadmap.md` for the phase inventory.

## Claude and Codex

Applies only when the Codex plugin is available.

- `small`: skip Codex unless risk is unusual.
- For any Codex invocation, **follow [`.claude/commands/use-codex.md`](.claude/commands/use-codex.md)**. That file is authoritative for this repo — it specifies the invocation path (Agent subagent vs Bash direct; the Skill path is banned), the operational rules (zombie check, concurrency, `--effort low` floor, gate off), and which command to reach for (`rescue` for docs/plans, `review` for standard diffs, `adversarial-review` for deep diffs).

Codex is a one-way critic. Do not assume a reverse loop exists.
Deep reference: [`.claude/docs/codex-usage-guide.md`](.claude/docs/codex-usage-guide.md).

**On capacity errors:** retry once. If still failing, proceed without Codex and log that it was skipped. Codex is valuable but never blocking — everything that invokes Codex must treat it as best-effort.

## Library Docs Lookup

Whenever you are unsure about a library, framework, SDK, API, CLI tool, or cloud service — its methods, signatures, config keys, version-specific behavior, or migration steps — **delegate to the `docs-researcher` subagent** instead of guessing from memory or reaching for web search. It is wired to the Context7 MCP server and runs on Haiku for speed and cost.

Use it for:

- "Does package X still expose method Y?" / "What are the valid args for Z?"
- Config schemas, env vars, CLI flags for tools we depend on.
- Version migrations (e.g. Pydantic v1→v2, LiteLLM config changes, Temporal SDK updates).
- Any API question where your training data may be stale.

Do **not** use it for: refactoring, writing scripts from scratch, debugging business logic, code review, or general programming concepts.

Invoke it via the Agent tool with `subagent_type: "docs-researcher"`. Give it the library name, the specific question, and — if relevant — the version pinned in the repo.

## Verification

No completion claims without fresh evidence.

1. Identify the command that proves the claim.
2. Run it.
3. Read the output and exit status.
4. Report the actual result.
5. Check `git status` before presenting completion.

Source of truth: `.claude/project/verification.md` and `.claude/project/invariants.md`.

## Learnings

- Add to `.claude/project/learnings.md` only after a verified, likely-to-recur pattern.
- Keep entries short. Never store secrets or machine-local paths.

## Scratchpad

- Use `scratchpad/` for any temporary files, caches, or throwaway work.
- This folder is gitignored and must never be committed.

## Git Safety

- Stage explicit files only. No `git add .`, `git add -A`, `--no-verify`, force-push, `reset --hard`, `clean`, `restore`, or `checkout` rewrites without explicit approval.
- Small reversible commits. Do not amend unless the user asks.
- Do not overwrite unrelated user changes.
- Do not encode machine-local absolute paths in plans, prompts, docs, or rules.
