---
name: component-review
description: Drive an exhaustive, manual, per-component review of the Bodha codebase for a given design version (e.g. v_2_7, v_2_7_v2). Maintains a persistent checklist at docs/reviews/<version>/index.md, fuses both architecture-trace outputs (at-claude + at-codex) with source code and design docs into a deeply-detailed review.md per component, runs Codex as a second-opinion critic, and pairs every review with a discussions.md for ongoing back-and-forth. Trigger when the user pairs a review verb — "review", "deep dive into", "walk me through", "continue the review pass", "next review", "let's review <component>", "/component-review" — with a Bodha component name or version. Also trigger when the user references docs/reviews/<version>/ paths, asks to resume the review pass, or signs off a component as complete. Do NOT trigger on a casual one-off question like "what does chitta do?" — that's a quick explanation, not a review session, and should not spawn a docs/reviews/ tree.
---

# Component Review

This skill is the manual-comprehension counterpart to `architecture-trace`. The arch-trace skill produces structured per-component documents under `docs/architecture-trace/at-claude/<V>/` and `docs/architecture-trace/at-codex/<V>/`. This skill takes those plus the source code and design docs, and walks the user through the system **one component at a time**, producing a deep `review.md` per component plus a paired `discussions.md` for ongoing dialogue.

## When this skill runs

- `/component-review <version>` (no component named) → bootstrap or resume the checklist for that version.
- `/component-review <version> <component>` → jump straight to that component.
- The user says things like "let's review dhriti", "next review", "continue the v_2_7_v2 review pass" → infer version from context (ask if ambiguous) and treat as a named or resume invocation.

`<version>` is a literal design tag like `v_2_7` or `v_2_7_v2`. There are no aliases. If the user typed `v1`/`v2`, ask which literal version they mean before continuing.

---

## Inputs the skill consumes per component

For component `<C>` at version `<V>`:

| Source | Path | Role |
|---|---|---|
| Claude arch-trace | `docs/architecture-trace/at-claude/<V>/01-<C>.md` | Primary structural understanding |
| Codex arch-trace | `docs/architecture-trace/at-codex/<V>/01-<C>.md` | Cross-check, surfaces blind spots |
| Cross-cutting (slice for `<C>`) | `docs/architecture-trace/at-claude/<V>/02-data-flow-full.md`, `02-dependency-matrix.md`, `02-contracts-and-protocols.md`, `02-temporal-workflows.md` | Inbound/outbound context |
| Design docs | `docs/design/v_2_7/core/` | Authoritative intent. `v_2_7_v2` has no separate design fork — always use `v_2_7/core/`. |
| Source code | `src/bodha/<C>/` (or `src/bodha/eval_harness/`, etc.) | Ground truth for current behavior |
| Existing user notes | `docs/reviews/<V>/<C>/discussions.md` if it exists | User's prior thoughts to fold in |

For `eval-harness`: the source lives outside `src/bodha/` proper — discover it (e.g. `src/bodha/eval_harness/` or a top-level `eval/`) before reading.

For dependencies: when the dependency-matrix shows `<C>` meaningfully depends on another component, **call out the dependency and link to its review.md** — do not inline a full sub-review. If that other component hasn't been reviewed yet, write a TODO link.

---

## Outputs the skill produces

```
docs/reviews/<V>/
  index.md                       # checklist (created on bootstrap, ticked manually)
  <component>/
    review.md                    # the deep review — 11 sections, see template
    discussions.md               # seeded header; user + Claude append over time
```

Both per-component files use the templates under [.claude/skills/component-review/templates/](templates/).

---

## Workflow

### Step 0 — Resolve version and bootstrap if needed

1. Parse the version from the invocation. If absent or ambiguous, ask the user.
2. Verify `docs/architecture-trace/at-claude/<V>/` exists. If it does not, stop and tell the user the arch-trace for that version hasn't been produced — direct them to `/architecture-trace` first.
3. Ensure `docs/reviews/<V>/` exists.
4. If `docs/reviews/<V>/index.md` is missing, **bootstrap it**:
   - Glob `docs/architecture-trace/at-claude/<V>/01-*.md` → derive the Components list (strip `01-` prefix and `.md` suffix).
   - Glob `docs/architecture-trace/at-claude/<V>/02-*.md` → derive the Cross-cutting topics list.
   - Render [templates/index.md.tmpl](templates/index.md.tmpl), substitute `<version>` and the two lists, write to `docs/reviews/<V>/index.md`.
   - Tell the user: "Bootstrapped index for `<V>` with N components and M cross-cutting topics."

### Step 0.5 — Detect interrupted work

Before picking a component in Step 1, scan the checklist for any `[~]` (in progress) entries. For each:

- If `docs/reviews/<V>/<C>/review.md` is missing → the prior session crashed before writing anything. Treat the entry as fresh `[ ]` and re-pick normally.
- If `review.md` exists but is clearly partial (sections empty, no Codex log) → tell the user, offer to resume from the next unfilled section, or restart from scratch.

This makes the `[~]` marker a real recovery signal, not just a status badge.

### Step 1 — Pick the component

- **User named one** (`/component-review v_2_7_v2 dhriti` or "let's review dhriti"): validate it appears in the checklist. If not, list valid items and stop.
- **User did not name one**: read `docs/reviews/<V>/index.md`, find the first item with status `[ ]` (todo) or `[~]` (in progress), propose it, and ask for confirmation before proceeding.
- **Already done** (`[x]`): tell the user this one is already reviewed; ask whether to skip to the next or "redo".
  - **Redo flow**: before rewriting `review.md`, append to `discussions.md`: `## <today's date> — Claude\nReview rewritten from scratch. Prior content above is preserved.` Then proceed. The discussions thread stays intact; only `review.md` is overwritten.

When you start working on a component, flip its checklist marker to `[~]` (in progress) so an interrupted session is resumable.

### Step 2 — Read the sources

Read in this order, taking notes as you go:

1. `docs/architecture-trace/at-claude/<V>/01-<C>.md` end-to-end. For `eval-harness` (or any component whose source is not under `src/bodha/<C>/`), this trace's header/summary names the actual source path — use it to locate the code in step 5.
2. `docs/architecture-trace/at-codex/<V>/01-<C>.md` end-to-end. Note any place the two traces disagree — this is gold for section 9 of the review. **If this file is missing**: record the gap in section 9, drop the cross-trace contradiction part of the Codex critic brief in Step 4, and proceed.
3. The 02-* files (`data-flow-full`, `dependency-matrix`, `contracts-and-protocols`, `temporal-workflows`). On first encounter per session, read each top-to-bottom — the headers and legends frame what the per-component rows actually mean. Grep'ing for `<C>` alone gives you matched lines without context.
4. `docs/design/v_2_7/core/` — read whichever design docs cover this component. Prefer `Glob` and `Grep` to locate the relevant files rather than reading everything.
5. `src/bodha/<C>/` (or the path identified in step 1 for off-tree components like `eval-harness`) — list the directory, read public surface (`__init__.py`, top-level modules, protocol/model definitions). Drill into specific files when sections 4 (How it works) or 5 (Public contracts) need ground-truth detail.
6. `docs/reviews/<V>/<C>/discussions.md` if it already exists — fold any user notes into the review.

If any source is missing or empty, note that gap explicitly in the review's section 9 and section 10. Do not silently skip.

### Step 3 — Draft `review.md`

Pick the template based on item type:

- **Component** (entries from the `## Components` section of the index): use [templates/review.md.tmpl](templates/review.md.tmpl) — 11 sections.
- **Cross-cutting topic** (entries from the `## Cross-cutting topics` section, e.g. `data-flow-full`, `dependency-matrix`): use [templates/review-crosscut.md.tmpl](templates/review-crosscut.md.tmpl) — 6 sections shaped for documents that describe relationships across components rather than one component's internals.

Conventions for both:

- Be exhaustive — this is the artifact the user will rely on without re-reading the source. Lean toward more detail, not less.
- **Visit every section.** Either fill it or write `N/A — <one-line reason>`. An empty header with no content is not acceptable; a short "N/A — this component has no Temporal workflows" is.
- Cite `file:line` for any non-obvious mechanic (lifecycle hooks, retry logic, state transitions, concurrency points).
- Use markdown links of the form `[name](src/bodha/<C>/file.py#L42)` so the user can click through.
- Quote design docs and arch-traces directly when summarising would lose nuance.
- If Claude's arch-trace and Codex's arch-trace disagree on a fact, **say so explicitly** in the contradictions section with both versions side by side.
- The Codex critique log section is filled in Step 4 — leave a placeholder for now.
- **Sources-read header must be honest.** The template ships with the full possible source list as a starting point; before writing, **prune it down to the files you actually opened**. If you didn't read `02-temporal-workflows.md` because the component doesn't use Temporal, drop that line. The Codex critic in Step 4 uses this header to anchor its checks — a stale list undermines the whole exercise.

Write the draft to `docs/reviews/<V>/<C>/review.md`.

### Step 4 — Codex critic pass

Run Codex via the rescue skill to critique the draft. Per AGENTS.md: standard scope uses `/codex:rescue --wait --fresh`. Brief Codex narrowly — the review's quality stands or falls on whether its claims about the actual code are true, so make that the primary check:

> Critique `docs/reviews/<V>/<C>/review.md`. Three checks, in order of priority:
> 1. **Code fidelity** — for every `file:line` citation in the review, open the file and verify the claim. Flag any citation that is wrong, stale, or describes the code inaccurately.
> 2. **Dependency completeness** — compare the review's "Dependencies" section against `docs/architecture-trace/at-claude/<V>/02-dependency-matrix.md`. Flag missing inbound or outbound edges.
> 3. **Cross-trace contradictions** — diff against `docs/architecture-trace/at-codex/<V>/01-<C>.md` (skip if missing). Flag facts where the two traces disagree and the review doesn't surface the disagreement.
>
> Do not rewrite the review — only flag issues, with `file:line` evidence for each finding.

Capture the critique. For each item:

- **Accept** → fix the review and note the change in section 11.
- **Dismiss** → record why in section 11.

On capacity errors: retry once. If still failing, proceed and log "Codex skipped: <reason>" in section 11. Codex is best-effort, not blocking.

### Step 5 — Seed `discussions.md` (only if missing)

If `docs/reviews/<V>/<C>/discussions.md` does not exist, write it from [templates/discussions.md.tmpl](templates/discussions.md.tmpl) with the component name and version filled in. If it exists, leave it alone — its content is sacred user notes.

### Step 6 — Hand off to the user, do **not** auto-tick

Surface the review to the user with a short summary of what was covered and the most interesting Codex findings. Then:

> "Review draft is at `docs/reviews/<V>/<C>/review.md`. Discussion thread is at `docs/reviews/<V>/<C>/discussions.md`. Want to discuss anything before I mark it complete?"

**Critical**: Do not flip the checklist marker to `[x]` on your own. The user explicitly drives completion — they may want to discuss, request edits, or push deeper into dependencies first. Discussion happens in `discussions.md` (append entries with date + speaker headers).

When the user says the component is done ("ok mark it complete", "done with dhriti", "tick it", etc.), the sign-off is written **by the skill on the user's command** — the user does not need to write the entry themselves first. The order is:

1. Append a sign-off entry to `discussions.md`: `## <today's date> — User\nMarked complete.` (Use whichever speaker tag is true — usually the user signing off, occasionally Claude noting "Marked complete per user".)
2. Update the line in `docs/reviews/<V>/index.md` from `[~]` to `[x]`.
3. Offer the next unreviewed item.

If the user later wants to revisit a `[x]` component, change it to `[!]` (needs revisit) before re-opening.

### Step 7 — Commit the new or updated files with a message like `docs: complete review of <C> for <V>`.
---

## Behavior rules

- **One component per invocation.** Do not auto-chain into the next component without the user saying so.
- **No subagents for reading or searching.** Do all Glob/Grep/Read work from the main agent. Subagents would paraphrase what they read and the review needs `file:line`-precise contact with the source. The only external delegation in this skill is the Codex critic in Step 4.
- **Read-only on source.** Never modify `src/bodha/`, `docs/design/`, or `docs/architecture-trace/`. The skill's writes are confined to `docs/reviews/<V>/`.
- **Templates are starting points, not straitjackets.** If a component genuinely has nothing to say in a section (e.g. a pure-utility module with no Temporal workflows), say "Not applicable — <reason>" rather than padding.
- **Cross-component links over inlining.** When `<C>` depends on `<D>`, link to `<D>`'s review (or write `[D — TODO](../<D>/review.md)` if not yet done). Keep each review focused.
- **Codex is a critic, not a co-author.** The review's voice stays Claude's. Codex's role is to find errors and gaps.
- **Discussions accumulate, never get rewritten.** When folding prior discussion notes into a redo, keep the original entries intact and append a new section noting the rewrite.

## Anti-patterns to avoid

- Writing a thin summary that just paraphrases the arch-trace. The arch-trace already exists; the review must add depth (mechanics, citations, contradictions, risks).
- Asking the user "should I proceed?" at every step. Be confident; only stop for the genuine decision points (which version, which component, mark complete).
- Skipping the Codex pass because it feels redundant. Even when Claude's draft looks complete, Codex routinely catches drift.
- Auto-ticking the checklist. The whole point of waiting is to leave room for discussion.

## Files in this skill

- [SKILL.md](SKILL.md) — this file
- [templates/index.md.tmpl](templates/index.md.tmpl) — bootstrap checklist
- [templates/review.md.tmpl](templates/review.md.tmpl) — 11-section review structure for components
- [templates/review-crosscut.md.tmpl](templates/review-crosscut.md.tmpl) — 6-section review structure for cross-cutting topics
- [templates/discussions.md.tmpl](templates/discussions.md.tmpl) — discussion seed header
