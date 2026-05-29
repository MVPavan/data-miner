---
description: Canonical invocation rules for the OpenAI Codex plugin in this repo. Subagents and the main thread must follow this when calling rescue, review, or adversarial-review.
---

# How to use Codex

Authoritative for this repo. Deep reference: [`.claude/docs/codex-usage-guide.md`](../docs/codex-usage-guide.md). On disagreement, this command wins.

## Invocation paths

| Command | Path | Fallback |
|---|---|---|
| `rescue` / `task` | Agent → `codex:codex-rescue` subagent | Bash → companion (capacity error or you are already a subagent — `agents.max_depth=1` blocks nested spawn) |
| `review` | Agent → general-purpose subagent (briefed with this file) | Bash → companion (same triggers as above) |
| `adversarial-review` | Agent → general-purpose subagent (briefed with this file) | Bash → companion (same triggers as above) |
| `status` / `result` / `cancel` / `setup` | Bash → companion | — |

**Skill tool path is banned for Codex.** Hangs on stale zombie state; pollutes main thread with verbose output.

**Bash for status/result/cancel/setup is required, not a fallback.** Plugin slash commands set `disable-model-invocation: true` (human UX, markdown-table rendering). Claude needs `node codex-companion.mjs <subcmd> --json` — machine-readable. Don't flip the flag; the split is intentional and keeps one code path.

## Rules

1. **Check zombies before any foreground call.** `status --all --json`; `cancel` any `running` job > 1h old (dead pid, unreconciled state). Uncancelled zombies silently deadlock the next foreground call.
2. **One heavyweight job at a time.** Never run two of `rescue` / `review` / `adversarial-review` concurrently — the shared-runtime broker starves one.
3. **`--effort` floor: `low`.** `minimal` is broken (rejects `web_search` → 400). `low` for small, `medium`/`high` for investigation, `xhigh` for deep root-cause. **Note:** `task --json` swallows the 400 body; if a job fails with empty `rawOutput`, read the `<logFile>` path from the job record for the real error.
4. **`reviewGateEnabled` stays `false`.** The stop-time gate can loop-burn usage limits. Verify with `setup --json`; flip with `setup --disable-review-gate --json`.
5. **Keep `agents.max_depth` at default `1`.** Nested subagents fan out exponentially.
6. **Prefer `--background` + polled `status --wait --timeout-ms` over foreground `--wait`** for review/adversarial-review. Foreground has no cancellation if the runtime hangs.
7. **Verify every Codex citation.** GPT-5.4 confidently cites lines that don't match current code.
8. **Pick by output shape:** `review` = prose; `adversarial-review` = structured JSON; `rescue` = free-form. `review` rejects focus text and `--scope staged|unstaged` — use `adversarial-review` if you need either.
9. **Iterate 2–5 rounds.** Exit on ship-it or only out-of-scope. Don't ship on "DO NOT SHIP" without fixing or documenting why it doesn't apply.
10. **Best-effort.** On capacity error: retry once, then proceed without it and log the skip.

## Invocation templates

**Rescue** — plugin subagent parses flags from prompt prefix:

```
Agent({
  subagent_type: "codex:codex-rescue",
  description: "<short label>",
  prompt: "--wait --fresh\n\n<self-contained brief>"
})
```

Forwards once and returns stdout verbatim. Will not read this file on its own — inline any shaping into the brief.

**Review / adversarial-review** — general-purpose Claude subagent absorbs Codex's verbose output:

```
Agent({
  subagent_type: "general-purpose",
  description: "Codex <adversarial-|>review of <target>",
  prompt: `
    Read .claude/commands/use-codex.md before acting.

    Use Bash with run_in_background: true (companion's --background returns
    a jobId quickly, but Bash still waits for that quick process unless
    detached at both layers):

      Bash({
        command: 'node "${CLAUDE_PLUGIN_ROOT}/scripts/codex-companion.mjs" \\
          <review|adversarial-review> --background \\
          --scope <auto|working-tree|branch> \\
          [--base <ref>] ["<focus, adversarial-review only>"] --json',
        run_in_background: true,
      })

    Poll: status <job-id> --wait --timeout-ms 900000 --json
    Fetch: result <job-id> --json
    Return findings verbatim. No paraphrase, no commentary.
  `
})
```

**Observability / lifecycle** — Bash direct, no wrapper:

```bash
COMPANION="${CLAUDE_PLUGIN_ROOT}/scripts/codex-companion.mjs"
node "$COMPANION" status --all --json                            # find zombies
node "$COMPANION" status <job-id> --wait --timeout-ms 900000 --json  # poll, 15-min cap
node "$COMPANION" cancel <job-id> --json                         # kill/cleanup
node "$COMPANION" result <job-id> --json                         # full payload
node "$COMPANION" setup --json                                   # readiness + gate
```

## Pointers

- Deep reference (architecture, gotchas, usage patterns, cheatsheet, fallback templates): [`.claude/docs/codex-usage-guide.md`](../docs/codex-usage-guide.md)
- Discussion history: [`.claude/docs/codex-discussions.md`](../docs/codex-discussions.md)
- Repo pointer: [`AGENTS.md`](../../AGENTS.md) § Claude and Codex
