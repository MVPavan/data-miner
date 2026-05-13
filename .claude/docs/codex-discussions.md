# Codex Usage — Discussions

> Staging doc. Captures live findings from the 2026-04-15 session, compares against prior-session learnings, and proposes a layered design. Not authoritative. Final version will land in `.claude/commands/use-codex.md` (the command) and `.claude/docs/codex-usage-guide.md` (the deep reference) once this thread closes.

---

## 1. Live findings from the 2026-04-15 session

All verified by direct execution against `codex-companion.mjs` and the Skill/Agent paths.

### 1.1 Invocation-path table (this session)

| Command | Bash → companion | Skill tool | Agent subagent |
|---|---|---|---|
| `setup` | ✅ verified | ✅ `codex:setup` verified | ❌ not registered |
| `status` | ✅ verified | ❌ not registered | ❌ not registered |
| `result` | ✅ verified | ❌ not registered | ❌ not registered |
| `cancel` | ✅ verified | ❌ not registered | ❌ not registered |
| `rescue` / `task` | ✅ verified | ⚠️ `codex:rescue` — hung 20+ min on first attempt, later worked | ✅ `codex:codex-rescue` verified (6m44s run) |
| `review` | ✅ verified (4m7s) | ❌ not registered | ❌ not registered |
| `adversarial-review` | ✅ verified (3m2s) | ❌ not registered | ❌ not registered |

ToolSearch over the available skill set confirms only `codex:rescue` and `codex:setup` are surfaced. No `codex:review` or `codex:adversarial-review` skill exists here.

### 1.2 Operational gotchas (verified live)

1. **`--effort minimal` is broken.** Codex always attaches the `web_search` tool, and the API rejects `web_search` at `effort=minimal`: `400 invalid_request_error`. Floor is `--effort low`. Verified on `task-mnzkwmdg-zkbe9z` (failed) and `task-mnzkx2fg-q97apm` (succeeded at `low`).
2. **Stale zombie jobs block Skill-path submissions.** A task from yesterday (`task-mnyjchr0-y85mrp`, pid 26770, dead for 17h) was still marked `running` in the companion's state file. That was the cause of the 20+ min hang on the Skill path this morning. `cancel <stale-id>` cleaned it up; subsequent Skill calls worked. The companion does not reconcile pid aliveness against job status.
3. **Shared-runtime broker starves parallel heavy jobs.** `sessionRuntime.mode = shared` at `unix:/tmp/cxc-lkkizq/broker.sock`. Running `review --wait` concurrently with `adversarial-review` caused the review to stall at 18m with only 4 tool calls. Running it alone afterwards completed in 4m7s. Rule: one heavyweight Codex job at a time.
4. **Standard `review` returns prose, not JSON.** Only `adversarial-review` returns the structured `verdict / findings[] / next_steps[]` schema documented at [`codex-usage-guide.md:205-227`](codex-usage-guide.md#L205-L227). The guide's claim (line 207) that "All reviews return structured JSON" is inaccurate for standard review — its output lives in `codex.stdout` as prose.
5. **`--background` + `status --wait --timeout-ms` is a safer pattern than foreground `--wait`.** Foreground `--wait` has no cancellation signal if the runtime hangs; background + polled status lets you check `progressPreview` and intervene.

### 1.3 Adversarial-review result against our own work

When the adversarial-review ran against the working tree, it caught a real substantive error in the contracts review §4.2: the rewrite claimed "LLM-facing contracts (`extraction.py`, `dharana.py`, `manas.py`, `skills.py`) give most fields permissive defaults so partial emissions still validate." That's false for the P3/P4 contracts — `CompressionResult.summary`, `EntityProfile.profile`, `HoldResolutionResult.verdict`, and several `MergePlan` / `VasanaCandidate` fields are still required. Partial or drifted model output on those boundaries will fail validation immediately. Confidence 0.93. Worth folding back into the review.

---

## 2. Prior-session learnings (from P5.0b Codex-heavy session, pasted by user)

### 2.1 What transfers verbatim (session-agnostic gold)

- **When to call**: before finalizing a design/plan, not during draft, not after commit. Codex is a critic, not a collaborator.
- **Iteration discipline**: 2-5 rounds for non-trivial work. Don't ship on a "DO NOT SHIP" without fixing the finding or documenting why it's out of scope.
- **Prompt shape**: context → what's decided → numbered specific questions → explicit out-of-scope → desired response shape.
- **Watch-outs**:
  - Codex confidently cites line numbers that don't match current code. Always verify.
  - When local shell is broken, Codex falls back to `origin/<branch>` via GitHub connector. Unpushed files show as "missing."
  - "DO NOT SHIP" can be overreach on out-of-scope concerns (budget policy, dormant-code test coverage). Triage: fix in-scope, document out-of-scope, push back.
  - Late-loop wording nits are often genuine factual errors in comments. Not pedantic.
- **Tool differentiation**: rescue for docs/plans/brainstorms; review for standard-risk code diffs; adversarial-review for deep-risk diffs (concurrency, data-loss, cross-cutting, migration).
- **The one rule**: loop until ship-it or only out-of-scope concerns remain.

### 2.2 What needs correction (session-dependent)

| Prior-session claim | This session's reality |
|---|---|
| "Skill `/codex:rescue` just routes to the same subagent — same destination with less indirection." | Skill path hung on a zombie state the subagent wasn't affected by. The claim is true **when the state is clean**; fragile when it isn't. |
| "For review / adversarial-review, use Skill: `Skill({ skill: \"codex:review\" })`." | No `codex:review` or `codex:adversarial-review` Skill exists in this session. Only Bash → companion works. |
| "Calling `codex-companion.mjs` directly skips capacity-handling, resume-candidate logic, and stdout normalization. I never invoked it that way." | Capacity / resume logic lives **inside** the CLI, not the wrappers. Wrappers fork the same binary. Bash direct is the only path for `status`, `result`, `cancel`, `setup`, `review`, `adversarial-review` in this session; dropping it would lose half the command surface. |
| "Both `review` and `adversarial-review` emit structured findings (BLOCKER / HIGH / MED / LOW)." | Only `adversarial-review` emits structured JSON. `review` emits prose. |

---

## 3. User's two clarifications (2026-04-15)

1. **Never use Skills if a subagent path exists.** Subagents give context isolation — Codex's long reasoning chain and tool-use trace stay out of the main thread. Skills pollute main context and have the zombie-state failure mode. If no subagent exists, Bash is the fallback.
2. **Feed the subagent the usage command.** Since the subagent will run Bash internally anyway, passing it a pointer to `/use-codex` (the command doc we're going to write) means the subagent knows how to invoke Codex correctly. The usage doc becomes an operationally live artifact consumed by subagents, humans, and the main agent equally.

---

## 4. Proposed synthesis — architecture

### 4.0 Two-layer architecture (terminology)

Every Codex invocation is two separate layers. Naming them precisely prevents confusion later.

```
┌─────────────────────────────────────────────────────────────┐
│ Main Claude thread (this conversation)                      │
│   uses Agent tool                                           │
└────────────────┬────────────────────────────────────────────┘
                 │ Agent({ subagent_type: "...", prompt: ... })
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1 — Claude-side subagent                              │
│   Model: Claude (separate instance, fresh context)          │
│   Harness: Claude Code / Claude Agent SDK                   │
│   Tools: Bash (and whatever else its definition grants)     │
│   Job: call the Codex CLI; absorb its verbose stdout;       │
│        return the distilled result to the main thread       │
└────────────────┬────────────────────────────────────────────┘
                 │ bash: node codex-companion.mjs <subcmd> ...
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2 — Codex itself                                      │
│   Model: GPT-5.4 (different model family)                   │
│   Harness: Codex CLI runtime, unrelated to Claude Code      │
│   Tools: Codex's own (file read, grep, web_search, etc.)    │
│   Job: do the actual critique / investigation               │
└─────────────────────────────────────────────────────────────┘
```

**Key points:**

1. **"Subagent" in this doc always means Layer 1 (Claude-side).** Codex itself is not a subagent of ours — it's a separate service reached via CLI fork.
2. **Context isolation happens at Layer 1.** The Claude-side subagent absorbs Codex's verbose reasoning chain + tool-use trace; only the distilled verdict crosses back to the main thread.
3. **Codex is indifferent to Layer 1.** It sees a CLI invocation with a prompt regardless of whether the caller is a Claude subagent, a direct Bash call, or a human at a terminal.
4. **`codex:codex-rescue` is a plugin-provided Layer-1 subagent.** Its definition at `plugins/.../agents/codex-rescue.md` is prescriptive:
   - *"Thin forwarding wrapper"* — one Bash call, return stdout verbatim, no commentary.
   - **Explicitly forbidden to call `review`, `adversarial-review`, `status`, `result`, or `cancel`** (line 27). It only forwards to `task`.
   - Cannot read `/use-codex` on its own — independent work is forbidden. To shape its behavior, guidance must be inlined in the `prompt` we pass it.
5. **For `review` / `adversarial-review`, no plugin subagent exists.** The workaround is a **general-purpose Layer-1 subagent** briefed with `/use-codex`, that runs Bash → companion internally and returns findings verbatim. Since the rescue subagent is type-locked to `task` only, a review-purpose subagent will never come from the plugin.

### 4.1 Architecture (revised with layer naming)

```
Main Claude thread (coordinator)
 │
 ├── rescue → Layer 1: codex:codex-rescue (plugin subagent)
 │            └── Layer 2: node codex-companion.mjs task
 │            └── returns verdict; main thread stays clean
 │
 ├── review / adversarial-review → Layer 1: general-purpose Claude subagent
 │            └── briefed with "read .claude/commands/use-codex.md, then run it"
 │            └── Layer 2: node codex-companion.mjs review|adversarial-review
 │            └── returns findings JSON verbatim
 │
 └── status / result / cancel / setup → Bash direct in main context
              └── outputs are small; no pollution risk
              └── ceremony of a subagent buys nothing here
```

### 4.1 Invocation patterns

**Rescue** (dedicated subagent exists):
```
Agent({
  subagent_type: "codex:codex-rescue",
  description: "short label",
  prompt: "--wait --fresh <context + questions>"
})
```

**Review / adversarial-review** (no dedicated subagent):
```
Agent({
  subagent_type: "general-purpose",
  description: "Codex adversarial review of diff",
  prompt: "Read .claude/commands/use-codex.md for invocation rules.
           Run: node <companion-path> adversarial-review --wait --scope working-tree '<focus>'
           Return the verdict + findings JSON verbatim. No paraphrasing."
})
```

**Status / result / cancel / setup** — Bash direct:
```
node <companion-path> status --all --json
```

### 4.2 When Bash direct is acceptable for rescue

Trivial ping-style questions where:
- Expected output is small (single line or short paragraph).
- The finding will not substantially influence main-context decisions.

The default is still subagent for any critique or investigation that produces findings-style output.

### 4.3 Rule set (the synthesis)

1. **Rescue**: subagent (`codex:codex-rescue`) primary, Bash fallback.
2. **Review / adversarial-review**: general-purpose Agent briefed with `/use-codex` primary, Bash fallback (main context absorbs output).
3. **Status / result / cancel / setup**: Bash direct.
4. **Skills are banned for Codex** — context pollution and zombie-state failure.
5. **Before any foreground invocation**: `status --all`; `cancel` any job `running` > few hours.
6. **Never run two heavyweight jobs concurrently** on the shared runtime.
7. **Effort floor is `low`** — `minimal` is broken.
8. **Iteration**: 2-5 rounds typical for non-trivial work. Exit on ship-it or out-of-scope-only.
9. **Verify citations before acting.** Codex hallucinates line numbers confidently.
10. **Standard review = prose; adversarial-review = structured JSON.** Pick the tool whose output shape you actually want.

---

## 5. The four-layer rollout plan

### Layer 1 — `.claude/docs/codex-usage-guide.md` (deep reference)

Updates:
- Fix `--effort minimal` entry (mark broken).
- Fix "All reviews return structured JSON" claim (only adversarial).
- Add an **Invocation Paths in Practice** section with the synthesis above.
- Point at Layer 3 as the day-to-day driver.

### Layer 2 — `.claude/project/learnings.md`

Two entries:
- "Codex: Skill path is banned — zombie state blocks it silently. Use subagent or Bash. See `.claude/commands/use-codex.md`."
- "Codex: one heavyweight job at a time on shared runtime; `--effort minimal` is broken (use `low` floor)."

### Layer 3 — `.claude/commands/use-codex.md` (new, authoritative)

Self-contained operational doc. Three sections:
- **(a) Invocation rules** — the table + the 10-rule set. Written so a subagent can read it cold and execute correctly.
- **(b) Usage patterns** — verbatim from the P5.0b learnings (when to call, prompt shape, watch-outs, ship-it rule).
- **(c) Subcommand cheatsheet** — flags per command, known error modes, the `minimal`-is-broken footnote.

### Layer 4 — `AGENTS.md`

Single-line update under "Claude and Codex": replace the three `/codex:rescue --wait --fresh` / `/codex:review --wait` / `/codex:adversarial-review --wait` lines with:

> For any Codex invocation, follow `.claude/commands/use-codex.md`.

Drop the current three prescriptive lines since they imply the Skill path.

---

## 6. Open questions to resolve before writing final

1. **Command name.** `/use-codex`? `/codex-how`? `/codex-rules`? Something else? Needs to be discoverable and not collide with plugin commands (`/codex:*` are taken).
2. **Layer 3 vs Layer 1 authority.** Current proposal: Layer 3 authoritative, Layer 1 deeper reference pointing at Layer 3. Alternative: keep Layer 1 authoritative, Layer 3 is a TL;DR pointer. Leaning toward the first because daily-driver commands actually get read.
3. **AGENTS.md scope.** Replace the three prescriptive lines wholesale, or keep them and add the pointer? Leaning wholesale replace — less surface area to keep in sync.
4. **Does `codex:codex-rescue` subagent read `/use-codex` on its own?** The subagent definition at `agents/codex-rescue.md` in the plugin is opaque to us. Either trust it as-is or belt-and-suspenders by passing the path in the prompt anyway. Leaning belt-and-suspenders.
5. **Trivial-rescue threshold.** Rule 1 says subagent primary, Bash fallback. Where's the trivial cutoff? Maybe: "if the expected output is < 5 lines AND won't influence main-context decisions, Bash direct is acceptable." Or just: "always subagent for rescue — ceremony is cheap."
6. **The pointer from `/use-codex` back to the deep guide.** Every section in Layer 3 that summarizes a concept should link to the corresponding section in Layer 1 for people who want the full story.

---

## 7. Parking lot (ideas raised but not yet decided)

- Whether to add a `codex:codex-review` or `codex:codex-adversarial-review` subagent definition to the repo's `.claude/agents/` — would remove the "general-purpose + brief" pattern and give us a first-class subagent path for all three review-like commands. Cost: repo-local subagent maintenance. Benefit: symmetric invocation pattern across all three tools.
- Whether to add a pre-invocation hook that runs `status --all | grep running > 4h` and warns if zombies are present. Would automate rule 5.
- Whether to document which `--effort` level to prefer per task type (extending the guide's existing table at line 396 with the corrections we've found).

---

_Last updated: 2026-04-15 · Authors: Claude (this session) + user (P5.0b prior-session learnings pasted inline)_
