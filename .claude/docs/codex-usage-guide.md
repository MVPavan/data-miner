# Codex Plugin Usage Guide for Claude Code

> **Day-to-day driver: [`.claude/commands/use-codex.md`](../commands/use-codex.md).** That file specifies the invocation path, operational rules, and command selection. This guide is the deep reference — schema detail, plugin internals, prompting blocks. Command wins on disagreement.

OpenAI Codex plugin from inside Claude Code. Codex runs **GPT-5.4** (different model family from Claude) for code review, adversarial analysis, diagnosis, implementation, and second opinions.

> **Source files** live at `${CLAUDE_PLUGIN_ROOT}` (resolves at plugin load time; concrete path on this machine: `/home/pavanmv/.claude/plugins/cache/openai-codex/codex/1.0.2/`):
>
> | What | Path (relative to plugin root) |
> |---|---|
> | Commands (all 7) | `commands/*.md` |
> | Prompt recipes & blocks | `skills/gpt-5-4-prompting/references/codex-prompt-recipes.md`, `prompt-blocks.md` |
> | Prompt anti-patterns | `skills/gpt-5-4-prompting/references/codex-prompt-antipatterns.md` |
> | Adversarial review prompt | `prompts/adversarial-review.md` |
> | Review output schema | `schemas/review-output.schema.json` |
> | Rescue agent definition | `agents/codex-rescue.md` |
> | Internal skills | `skills/{codex-cli-runtime,codex-result-handling,gpt-5-4-prompting}/` |
> | Stop-time review gate | `prompts/stop-review-gate.md` |
> | Hooks config | `hooks/hooks.json` |
> | Runtime script | `scripts/codex-companion.mjs` |

---

## Quick Reference

| Command | Use Case | Key Flags |
|---|---|---|
| `/codex:review` | Standard code review (prose) | `--wait`, `--background`, `--base`, `--scope` |
| `/codex:adversarial-review` | Challenge design/approach (structured JSON) | Same + `[focus text]` |
| `/codex:rescue` | Debug / fix / research (free-form) | `--model`, `--effort`, `--resume`, `--fresh` |
| `/codex:status` | Monitor jobs | `--wait`, `--timeout-ms`, `--all` |
| `/codex:result` | Retrieve completed job output | `[job-id]` |
| `/codex:cancel` | Stop a job | `[job-id]` |
| `/codex:setup` | Readiness, toggle review gate | `--enable-review-gate`, `--disable-review-gate` |

---

## Invocation Paths in Practice (repo policy)

Authoritative rules in [`.claude/commands/use-codex.md`](../commands/use-codex.md). Summary:

- **`rescue` / `task`** → Agent → `codex:codex-rescue` subagent. Bash fallback on capacity error **or** when the caller is itself a subagent (`agents.max_depth=1` blocks nested spawn).
- **`review` / `adversarial-review`** → Agent → general-purpose Claude-side subagent briefed with `/use-codex`. Bash fallback under the same triggers.
- **`status` / `result` / `cancel` / `setup`** → Bash direct in main thread.
- **Skill tool path is banned** — zombie state hangs + main-thread context pollution.

### Two layers

Every Codex invocation:

1. **Layer 1 — Claude-side subagent.** Separate Claude instance (Agent tool). Absorbs Codex's reasoning + tool-use trace; returns distilled result.
2. **Layer 2 — Codex itself.** GPT-5.4 in Codex CLI runtime, reached via `node codex-companion.mjs <subcmd>`. Unrelated to Claude Code.

`codex:codex-rescue` is a Layer-1 subagent locked to `task` only (see `agents/codex-rescue.md`). For `review` / `adversarial-review` no plugin subagent exists — use general-purpose and inline guidance.

### Operational gotchas (verified live 2026-04-15)

1. **Zombie jobs block foreground.** Dead pid + state still `running` → next foreground call hangs. `status --all`, `cancel` any job > 1h old.
2. **Shared-runtime broker starves parallels.** `sessionRuntime.mode = shared`. Two heavy jobs concurrently → one stalls 15+ min. Serialize.
3. **`--effort minimal` is broken** (web_search incompatibility, see Quick Reference correction). Companion bug: `task --json` with bad effort returns exit 1 and `rawOutput: ""` — the actual 400 message lives only in the `<logFile>` path on the job record. Read that file when JSON output is empty.
4. **Review gate burns usage limits.** Keep `reviewGateEnabled: false`; verify with `setup --json` at session start.
5. **`agents.max_depth = 1`.** Nested Claude-side subagents → exponential Codex fan-out.
6. **Codex citations can be wrong.** Always grep/read the cited file before acting.
7. **`--background` + polled `status --wait --timeout-ms`** beats foreground `--wait` for review/adversarial-review — foreground has no cancellation if runtime hangs.

### Templates

Listed for reference; `/use-codex` is authoritative.

**Rescue:**

```
Agent({
  subagent_type: "codex:codex-rescue",
  description: "<short label>",
  prompt: "--wait --fresh\n\n<brief>"
})
```

Subagent parses `--wait`/`--fresh`/`--resume`/`--background` from prefix. Won't read `/use-codex` on its own — inline guidance.

**Review / adversarial-review:**

```
Agent({
  subagent_type: "general-purpose",
  description: "Codex <adversarial-|>review of <target>",
  prompt: `
    Read .claude/commands/use-codex.md.
    Bash({ command: 'node ".../codex-companion.mjs" <review|adversarial-review> \\
      --background --scope <auto|working-tree|branch> \\
      [--base <ref>] ["<focus, adv only>"] --json',
      run_in_background: true })
    Poll: status <id> --wait --timeout-ms 900000.
    Fetch: result <id>. Return findings verbatim.
  `
})
```

**Observability:**

```bash
COMPANION="${CLAUDE_PLUGIN_ROOT}/scripts/codex-companion.mjs"
node "$COMPANION" status --all --json                            # all jobs incl zombies
node "$COMPANION" status <id> --json                             # one-job detail
node "$COMPANION" status <id> --wait --timeout-ms 900000 --json  # poll
node "$COMPANION" result <id> --json                             # full payload
node "$COMPANION" cancel <id> --json                             # kill/cleanup
node "$COMPANION" setup --json                                   # readiness + gate
node "$COMPANION" setup --disable-review-gate --json             # restore default
```

**Bash fallback for rescue/review/adv-review** (subagent capacity error or plugin missing):

```bash
node "$COMPANION" task --background --effort low --fresh "<prompt>" --json
node "$COMPANION" review --background --scope working-tree --json
node "$COMPANION" adversarial-review --background --scope working-tree "<focus>" --json
```

Bash fallback puts Codex's full output into the main thread — retry the subagent path once capacity recovers.

### Double-detachment

Companion's `--background` queues the job and returns a `jobId` quickly, but the Bash tool waits for the (quick) companion process unless you also pass `run_in_background: true`. For small cases this is invisible; for robustness, pass both:

```
Bash({
  command: 'node "$COMPANION" review --background --scope working-tree --json',
  run_in_background: true
})
```

Plugin's own `review.md` and `adversarial-review.md` document this dual-flag requirement.

### Pre-flight size estimation

Before picking foreground vs background, the plugin commands estimate review size. Reproduce when scripting:

```bash
# working-tree scope
git status --short --untracked-files=all
git diff --shortstat --cached       # staged
git diff --shortstat                # unstaged

# branch scope
git diff --shortstat <base>...HEAD
```

Plugin rule: `--wait` only for clearly tiny diffs (~1–2 files). Otherwise `--background`. Untracked files count as reviewable even with empty `git diff --shortstat`.

### Resume helper

```bash
node "$COMPANION" task-resume-candidate --json
```

Returns `{ available: bool, ... }`. `true` → prior rescue thread can be resumed via `--resume` / `--resume-last`. Use this to choose `--fresh` vs `--resume` instead of guessing.

### Scope restrictions

- **`/codex:review`** — supports `--scope auto|working-tree|branch` and `--base <ref>`. Rejects `--scope staged`, `--scope unstaged`, and any focus text. *"If you need custom review instructions or adversarial framing, use `/codex:adversarial-review`."*
- **`/codex:adversarial-review`** — same scope options, **plus** free-form focus text after flags (e.g. `"race conditions and data loss"`). Still rejects staged/unstaged.

### Model aliases

Default: GPT-5.4. Set explicitly only with reason.

- `spark` → `gpt-5.3-codex-spark` (lighter/faster; plugin maps the alias).
- `gpt-5.4-mini`, `gpt-5.3-mini` — pass through literally.

Pairs with `--effort` (floor `low`; `minimal` broken).

### Setup states

1. **Installed + authenticated** → `ready: true`.
2. **npm available, Codex missing** → `/codex:setup` offers `npm install -g @openai/codex`. Scripted setup: install yourself then rerun `setup --json`.
3. **Installed, unauthenticated** → `auth.loggedIn: false`. User runs `codex login` from a terminal (browser flow, not scriptable from inside Claude Code).

### Usage patterns (when and how)

Originally captured during a Codex-heavy session. Codex users converge on this independently.

**When to call:**

- **Before finalizing a design / plan** — not during draft, not after commit. Tell Codex "critique what's decided, not help me decide."
- **Iteratively** — 2–5 rounds for non-trivial work; each round surfaces new layers as prior ones resolve.
- **As a critic, not a collaborator** — Codex doesn't see the conversation or the local shell. Feed it the artifact, the decision, the specific doubt.

**Prompt shape:**

- **Context** — what the system is, why the change exists.
- **What's already decided and patched** — so Codex doesn't re-litigate closed loops.
- **Numbered specific questions** — not "review this." Ask "does X bind Y correctly?", "is Z's fallback what I think?"
- **Out of scope** — prevents scope drift in the response.
- **Desired response shape** — "2-line ship-it or exact line + exact fix." Without this, Codex writes paragraphs.

**Execution flags (rescue):**

- `--wait --fresh` — default for new topic. Blocks until done; ignores prior thread.
- `--resume` — only for genuinely continuous follow-ups. Else fresh is cleaner.
- Avoid `--background` for rescue unless you have parallel work — the next action usually depends on the verdict.

**Watch-outs:**

- **No working-tree visibility when shell is broken** — Codex falls back to `origin/<branch>` via GitHub connector. Unpushed files show as missing. Verify locally before acting on a FAIL that hinges on a known-present file.
- **Invented citations** — Codex confidently cites line numbers. Always grep the claim.
- **"DO NOT SHIP" overreach** — may flag genuinely out-of-scope concerns (budget policy, dormant-code coverage). Triage: fix in-scope, document out-of-scope in the plan, push back.
- **Late-loop wording nits** — once substantive issues close, Codex finds genuine semantic traps in comments. Reads as pedantic but usually catches real factual errors.

**Good at:** cross-checking code against named spec docs · stale comments · path hallucinations · missing test files · pushing back on assumptions disguised as facts.

**Not good at:** writing for you (treat as read-only even with `--write`) · open-ended "what should I do" (sharpen to "given these decisions, what breaks first") · trusting plan files exist on disk (always say where, or paste short docs inline).

**Tool selection:**

- **Pure doc work** (brainstorm, plan, roadmap) → `rescue`. `review` has nothing to operate on.
- **Small/standard code change** → `review`. One pass, defect-focused, prose.
- **Deep code change** (cross-cutting, concurrency, migrations, security) → `adversarial-review`. Structured findings; tries to break it.
- **Stuck / second diagnosis** → `rescue --fresh` with the symptom.

**The one rule:** loop until ship-it or only out-of-scope concerns. Don't ship on "DO NOT SHIP" without fixing the finding or writing why it doesn't apply.

---

## Subcommand Detail

`COMPANION="${CLAUDE_PLUGIN_ROOT}/scripts/codex-companion.mjs"`

### `setup`

Bash direct. Readiness check (Node, npm, Codex CLI, auth, shared-runtime endpoint, gate status).

- Flags: `--enable-review-gate`, `--disable-review-gate`, `--json`.
- Verify `reviewGateEnabled: false` each new session.
- Auto-install + auth fallback states above.

### `status`

Bash direct. List jobs / inspect one / poll until complete.

- Flags: `[<id>]`, `--all`, `--wait`, `--timeout-ms <ms>`, `--json`. Default polling timeout 4 min.
- No id → table of current + past runs. With id → full detail.
- Known error mode: no pid/state reconciliation — dead pid with `status=running` needs manual `cancel`.

### `result`

Bash direct. Full payload of a completed job.

- Output by command:
  - `review` → prose in `storedJob.result.codex.stdout` and `storedJob.rendered`.
  - `adversarial-review` → structured JSON in same field, matches `review-output.schema.json` (see Schema below).
  - `task`/`rescue` → `storedJob.result.rawOutput` (free-form text); `threadId` for `--resume-last`.

### `cancel`

Bash direct. Stop a running job or clean up a zombie.

- Returns `{turnInterruptAttempted, turnInterrupted}`. `turnInterruptAttempted: false` = job past interruptible window (state still flips to `cancelled`).
- Safe against zombies.

### `task` (alias `rescue`)

Agent subagent primary; Bash fallback. Workhorse delegation to GPT-5.4. Codex can read and write files in the repo.

- Flags (parsed from prompt prefix by subagent): `--wait`, `--background`, `--fresh`, `--resume` / `--resume-last`, `--effort`, `--model`, `--write`. Subagent defaults to `--write`; remove explicitly if read-only.
- Effort: `none` | ~~`minimal`~~ | `low` | `medium` | `high` | `xhigh`. **`minimal` broken** (web_search rejected at minimal → 400). Floor `low`.
- Models: default GPT-5.4; `spark` → `gpt-5.3-codex-spark`; `gpt-5.4-mini`, `gpt-5.3-mini` literal.
- Thread: omit `--resume`/`--fresh` and the slash command asks. Keywords for resume: continue, keep going, resume, apply the fix, dig deeper.
- When to use: stuck on a problem, want second opinion / deeper investigation, substantial debug or impl benefitting from GPT-5.4, long tool-heavy work.
- When NOT to use: simple tasks Claude handles directly, quick questions/explanations.
- Known error modes: `--effort minimal` → 400; zombie state → silent foreground hang.

Examples:
```
/codex:rescue Debug why the test suite times out after the extraction refactor
/codex:rescue --background --effort high Fix the projection_processor concurrency bug
/codex:rescue --resume Apply the top three fixes from your analysis
```

### `review`

Agent subagent primary; Bash fallback. Standard quality review of the diff.

- Flags: `--wait` | `--background`, `--base <ref>` (default `main`), `--scope auto|working-tree|branch`.
- No `--scope staged|unstaged`, no focus text — use `adversarial-review` instead.
- Output: prose (not JSON).
- If neither `--wait` nor `--background`, the slash command estimates size (see Pre-flight) and asks.

Examples:
```
/codex:review --wait
/codex:review --background --base main
/codex:review --scope working-tree
```

### `adversarial-review`

Agent subagent primary; Bash fallback. Skeptical review — actively tries to break the change.

- Flags: same as `review`, plus `[focus ...]` text.
- Output: structured JSON (see Schema). `verdict ∈ {approve, needs-attention}`.
- Typical duration alone on shared runtime: 3–5 min.
- Attack surface: auth/permissions/tenant isolation/trust boundaries, data loss/corruption/duplication/irreversible state, rollback safety/retries/partial failure/idempotency, races/ordering/stale state/re-entrancy, empty-state/null/timeout/degraded deps, version skew/schema drift/migration hazards/compat regressions, observability gaps.
- Method: tries to disprove the change. Looks for violated invariants, missing guards, unhandled failure paths. Traces bad inputs, retries, concurrent actions, partial operations.

Examples:
```
/codex:adversarial-review --wait "concurrent access patterns"
/codex:adversarial-review --background --base develop "auth and data loss"
```

---

## Review Output Schema

`review` returns prose in `codex.stdout` (markdown copy in `storedJob.rendered`); does **not** emit the schema.

`adversarial-review` returns structured JSON:

```json
{
  "verdict": "approve" | "needs-attention",
  "summary": "Terse ship/no-ship assessment",
  "findings": [
    {
      "severity": "critical | high | medium | low",
      "title": "Short finding title",
      "body": "What can go wrong, why vulnerable, impact, concrete fix",
      "file": "path/to/file.py",
      "line_start": 42,
      "line_end": 55,
      "confidence": 0.85,
      "recommendation": "Concrete fix suggestion"
    }
  ],
  "next_steps": ["Action item 1", "Action item 2"]
}
```

Each finding answers: (1) what can go wrong? (2) why is the path vulnerable? (3) likely impact? (4) concrete change that reduces risk?

---

## Prompting Codex Effectively

GPT-5.4 responds best to XML-tagged prompts.

```xml
<task>Define the concrete job and expected end state. Be specific —
"Review this change for correctness" beats "Take a look."</task>

<structured_output_contract>
1. Root cause  2. Evidence  3. Recommended fix  4. Residual risks
</structured_output_contract>

<default_follow_through_policy>Keep going until you have enough evidence
to identify the root cause confidently. Do not stop at the first plausible
answer.</default_follow_through_policy>

<verification_loop>Before finalizing, verify the proposed solution matches
the observed evidence and doesn't introduce new issues.</verification_loop>

<grounding_rules>Ground every claim in repository context or tool outputs.
Label inferences explicitly.</grounding_rules>
```

**Available blocks:**

- *Output:* `<structured_output_contract>` (schema), `<compact_output_contract>` (concise prose).
- *Follow-through:* `<default_follow_through_policy>`, `<completeness_contract>`, `<verification_loop>`.
- *Grounding:* `<missing_context_gating>` (don't guess; retrieve), `<grounding_rules>`, `<citation_rules>`.
- *Safety:* `<action_safety>` (tightly scoped, no unrelated refactors), `<tool_persistence_rules>`.
- *Task:* `<research_mode>` (separate facts/inferences/open questions), `<dig_deeper_nudge>` (second-order failures), `<progress_updates>`.

---

## Recipe Templates

**Diagnosis:**
```
/codex:rescue --effort high Diagnose why the extraction buffer flush is dropping memories.
Focus on the timing between Phase 1 completion and the flush trigger check.
```
Returns: root cause, evidence, smallest safe next step.

**Narrow fix:**
```
/codex:rescue --effort medium Fix the race condition in projection_processor where
concurrent entity updates produce duplicate Neo4j nodes. Preserve existing behavior
outside the failing path.
```
Returns: fix summary, touched files, verification, residual risks.

**Root-cause analysis:**
```
/codex:rescue --background --effort xhigh Investigate why temporal contradiction detection
misses contradictions when the superseding memory arrives within the same extraction batch.
Trace from extraction buffer flush through Phase 2 verdicts.
```
Returns: findings ordered by severity, evidence, next steps.

**Research / recommendation:**
```
/codex:rescue --effort high Research whether to use GDS community detection or a custom
clustering approach for memory consolidation. Consider graph size (~50K nodes), update
frequency, deterministic results requirement.
```
Returns: facts, recommendation, tradeoffs, open questions.

---

## Anti-Patterns to Avoid

| Anti-pattern | Problem | Better |
|---|---|---|
| Vague framing | "Take a look at this" | "Review this change for correctness and regression risks" |
| No output contract | "Investigate and report back" | Specify what the output should contain |
| No follow-through | "Debug this failure" | "Keep going until you have enough evidence" |
| Asking for more reasoning | "Think harder, be very smart" | Add a `<verification_loop>` instead |
| Mixed jobs | "Review, fix, update docs, suggest roadmap" | One task per `/codex:rescue` |
| Unsupported certainty | "Tell me exactly why prod failed" | "Ground every claim in context" |

---

## Using Codex with Bodha Skills

**`bodha-memory-eval` cross-validation** — multi-model diversity (Opus + GPT-5.4 agree → high confidence):
```
/codex:rescue --effort high --wait Review these scenario evaluation findings for accuracy.
Check whether cited design doc sections actually support the claims.
Scenario: [paste]. Findings: [paste].
```

**Design document review** (after `design-evolve`):
```
/codex:adversarial-review --background --base main "design consistency and invariant violations"
```

**Code implementation review** (Bodha components):
```
/codex:adversarial-review --wait "race conditions, data loss, temporal consistency"
```

---

## Model & Effort Selection Guide

| Task | Model | Effort |
|---|---|---|
| Quick sanity check | `spark` | `low` |
| Standard code review | default | (use `/codex:review`) |
| Adversarial review | default | (use `/codex:adversarial-review`) |
| Bug diagnosis | default | `high` |
| Complex root-cause | default | `xhigh` |
| Implementation/fix | default | `medium`–`high` |
| Research/recommendation | default | `high` |
| Follow-up/resume | default | `medium` |

---

## Job Lifecycle

```
Start ──→ Running ──→ Completed ──→ result
                │
                └──→ Cancelled

Foreground (--wait):  block until done, get result
Background:           jobId → status → result
```

**Default timeouts:** status polling 4 min (`--timeout-ms`); review gate 15 min; session hooks 5 s each.

---

## Stop-Time Review Gate

Optional safety net: runs a Codex review before session end.

- Enable: `/codex:setup --enable-review-gate`. Disable: `/codex:setup --disable-review-gate`.
- When enabled, `/stop` triggers an automatic review of latest changes. Only direct code edits; status checks and command outputs skipped. Returns `ALLOW` (safe) or `BLOCK` (fix first).
- Checks for second-order issues: empty-state, retries, stale state, rollback risk, design tradeoffs not obvious from the diff.
- **Repo policy: keep this `false`** (see Operational gotchas — gate can loop-burn usage limits).
