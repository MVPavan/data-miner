# Durable Learnings

Add entries here only after a fix or pattern has been verified.

## Entry Format

### YYYY-MM-DD - Short Title

- Scope:
- Trigger:
- Rule:
- Evidence:
- Related docs:

---

### 2026-04-15 — Codex invocation: subagent for critique work, Bash for observability, never Skills

- Scope: any time Claude Code invokes the OpenAI Codex plugin (`rescue`, `review`, `adversarial-review`, `status`, `result`, `cancel`, `setup`).
- Trigger: verifying why a `codex:rescue` Skill call hung 20+ minutes without consuming Codex tokens; comparing Skill vs Agent vs Bash invocation paths live.
- Rule: `rescue` → Agent tool → `codex:codex-rescue` subagent; `review` / `adversarial-review` → Agent tool → general-purpose subagent briefed with `.claude/commands/use-codex.md`; `status` / `result` / `cancel` / `setup` → Bash direct. **Skill path is banned**: it silently hangs on stale zombie-job state in the companion's state file (dead pid still marked `running`) and pollutes the main thread with Codex's verbose output. Fallback to Bash-direct for rescue/review only when the subagent hits a capacity error.
- Evidence: (a) live session 2026-04-15 — Skill `codex:rescue` hung while zombie `task-mnyjchr0-y85mrp` (pid 26770, dead 17h) was marked `running`; manual `cancel` unblocked it. (b) ToolSearch confirmed no `codex:review` or `codex:adversarial-review` Skill exists in this session — only Bash direct or a general-purpose subagent wrapper works. (c) Subagent definition at `plugins/cache/openai-codex/codex/1.0.2/agents/codex-rescue.md` is type-locked to `task` only.
- Related docs: `.claude/commands/use-codex.md` (authoritative rules), `.claude/docs/codex-usage-guide.md` (deep reference), `.claude/docs/codex-discussions.md` (design history).

### 2026-04-15 — Codex operational floor: --effort low, one heavyweight job at a time, gate off

- Scope: every Codex CLI invocation via `codex-companion.mjs`.
- Trigger: systematic sanity check of all seven Codex subcommands with varied params against a live runtime.
- Rule: (1) **`--effort minimal` is broken** — Codex always attaches `web_search` which is rejected at `minimal` (`400 invalid_request_error`). Floor is `low`; `none` also works. (2) **One heavyweight job at a time** — the shared-runtime broker (`sessionRuntime.mode = shared`) starves parallel review / adversarial-review jobs (observed an 18m stall in a review running concurrently with an adversarial-review). Serialize. (3) **`reviewGateEnabled` stays `false`** — it can loop-burn usage limits (community-reported pitfall, matches design). Verify with `setup --json` each session; run `setup --disable-review-gate --json` if true. (4) **`review` returns prose, not JSON** — only `adversarial-review` returns the structured `verdict / findings[] / next_steps[]` schema; pick the command whose output shape you want.
- Evidence: (a) job `task-mnzkwmdg-zkbe9z` failed with `"tools cannot be used with reasoning.effort 'minimal': web_search"`; retry at `--effort low` (`task-mnzkx2fg-q97apm`) completed in 3s. (b) Parallel runs: `review-mnzkynd9-hq2l6j` stalled at 18m+ with 4 progress events while `review-mnzla62j-lzwcl0` ran; isolated rerun of review completed in 4m7s. (c) `result <id> --json` for standard review shows prose in `storedJob.result.codex.stdout` vs structured JSON for adversarial.
- Related docs: `.claude/commands/use-codex.md` §A.3 (rules 5–7) and §C (per-subcommand output shape).

---

### 2026-04-10 — pydantic-ai 1.x ModelRequestContext is a dataclass, not a BaseModel

- Scope: any capability hook that mutates the outgoing model request (`before_model_request`, `wrap_model_request`, `after_model_request` return value).
- Trigger: writing C1 (ContextRepackCapability) correction and asking "does my mutation path actually reach the next LLM call?"
- Rule: `pydantic_ai.models.ModelRequestContext` is a non-frozen `@dataclass(kw_only=True)` with fields `model`, `messages`, `model_settings`, `model_request_parameters`. It has **no `.model_copy()`**. To return a mutated context, use `dataclasses.replace(request_context, messages=...)` — NOT `request_context.model_copy(update={...})`. The docstring and type-checker both obscure this because `ModelRequestContext` sits in a Pydantic-flavored library but is plain stdlib.
- Evidence: (a) source read at `.venv/lib/python3.12/site-packages/pydantic_ai/models/__init__.py`, (b) empirical spike at `scratchpad/spikes/repack_mutation_test.py` — first run crashed with `AttributeError: 'ModelRequestContext' object has no attribute 'model_copy'`; second run after the fix produced "Berlin" when the sentinel swapped France→Germany.
- Related docs: §0.1 "Correction 2026-04-10b — C1-fix" in `docs/design/v_2_7/core/new_discussions/bodha-pydanticai-temporal-litellm-integration.md`; P5.5/P5.6/P5.7 in `docs/roadmap-pydanticai-integration.md`.

### 2026-04-10 — pydantic-ai capability hooks do not fire after message_history is updated

- Scope: any capability that needs to react to the updated message history after a tool executed.
- Trigger: designing ContextRepackCapability to repack memory after `recall_memory` tool returns.
- Rule: `after_tool_execute` fires *before* the tool result is appended to message history. History mutation happens in `_handle_final_result` at `_agent_graph.py:1204`, after all hooks return. There is no hook that runs after history update. If a capability needs the updated history, move the work to the next `before_model_request` and gate it on "last message is a tool-result part". Add per-run idempotency markers (set keyed on `(len(messages), last_tool_call_id)`, cleared by `for_run`) to handle pydantic-ai retry loops that re-see the same batch.
- Evidence: source read at `.venv/lib/python3.12/site-packages/pydantic_ai/_agent_graph.py:1204`, `capabilities/abstract.py:410-548`. Codex critique 2026-04-10.
- Related docs: §0.1 C1 + C1-idempotency in the integration brainstorm; P5.5 in the roadmap.

### 2026-04-10 — pydantic-ai capability hook ordering is symmetric middleware

- Scope: writing `CombinedCapability` with multiple children that need deterministic ordering.
- Trigger: writing Bodha's 5-capability chain (Turn → Budget → Repack → ECB → Observability).
- Rule: `before_*` hooks fire in registration order. `after_*` hooks fire in **reverse** order. `wrap_*` hooks nest middleware-style with the first capability as outermost. This is the standard middleware pattern — state-mutating capabilities should be registered first so they see requests earliest and responses latest. Also: `CombinedCapability.for_run` runs child `for_run` calls under `asyncio.gather`. If `for_run` is truly pure construction, concurrency buys nothing; if it ever isn't, `gather` makes bugs nondeterministic. Subclass `CombinedCapability` with a sequential `for`-loop `for_run` for safety.
- Evidence: `.venv/lib/python3.12/site-packages/pydantic_ai/capabilities/combined.py:47-51` (gather), pydantic-ai docs explicitly state symmetric-middleware ordering.
- Related docs: §0 C3 + §0.1 C4-sequential in the integration brainstorm; P5.8 in the roadmap.

### 2026-04-10 — pydantic-ai TemporalAgent serializes deps via Pydantic TypeAdapter; use PrivateAttr + lazy @property for live clients

- Scope: passing non-serializable infrastructure (Redis client, httpx client, connection pools) as deps to a `TemporalAgent`-wrapped agent.
- Trigger: writing `BodhaAgentDeps` and worrying about serialization of live clients across the activity boundary.
- Rule: TemporalAgent uses `temporalio.contrib.pydantic.PydanticPayloadConverter`, which serializes deps via Pydantic `TypeAdapter`. Non-serializable fields raise `PydanticSerializationError` wrapped as `UserError` at `_agent.py:294-297`. **Solution:** one frozen Pydantic `BaseModel` with regular fields for serializable config (IDs, URLs, enums) and `PrivateAttr(default=None)` for live clients. Live clients are exposed via `@property` that (a) checks `temporalio.workflow.in_workflow()` and raises `RuntimeError("activity-only")` if true, and (b) looks up a worker-local pool seeded at `@worker.on_start`. `PrivateAttr` fields are automatically excluded from `TypeAdapter` serialization — confirmed empirically via 122-byte round-trip in `scratchpad/spikes/pydanticai_full_stack_spike.py`. No two-class split, no custom `TemporalRunContext` subclass needed in the common case.
- Evidence: (a) source `.venv/lib/python3.12/site-packages/pydantic_ai/durable_exec/temporal/__init__.py:9,44-61` (PydanticPayloadConverter), `_agent.py:294-297` (error wrap), `_model.py:156-163` (deps passed as activity arg). (b) spike round-trip verified. (c) Codex critique 2026-04-10 on hardening the workflow-context guard.
- Related docs: §0 C5 + §0.1 C5-hard-fail + C5-lifecycle in the integration brainstorm; P1.3, P1.3b in the roadmap.

### 2026-04-10 — pydantic-ai TemporalAgent activity names use `agent__{name}__*` format with double underscores

- Scope: committing to Temporal activity names that must survive refactors without breaking in-flight workflows.
- Trigger: writing the activity naming freeze document in P3.4.
- Rule: `TemporalAgent(agent, name='<name>')` constructs activities with the prefix `f'agent__{self.name}'` at `_agent.py:144`, then builds names like `agent__<name>__model_request`, `agent__<name>__model_request_stream`, `agent__<name>__toolset__<id>__call_tool`, `agent__<name>__event_stream_handler`, `agent__<name>__dynamic_toolset__<id>__get_tools`. Separators are all double underscores. Colons, dots, and other separators do not work. Once deployed, renaming the `name` parameter is breaking for any in-flight workflow because Temporal routes by activity name.
- Evidence: source reads at `_agent.py:144`, `_model.py:84,123`, `_function_toolset.py:61`, `_dynamic_toolset.py:77,102`; spike introspection at `scratchpad/spikes/pydanticai_full_stack_spike.py` discovered `agent__bodha-spike__event_stream_handler`.
- Related docs: §0 C6 in the integration brainstorm; P2.4, P3.4 in the roadmap.

### 2026-04-10 — pydantic-ai 1.x renames OpenAIModel to OpenAIChatModel + OpenAIProvider split

- Scope: constructing a pydantic-ai model that points at an OpenAI-compatible endpoint (LiteLLM proxy, vLLM, Ollama, etc.).
- Trigger: writing `ModelGateway.create_model(capability)` factory.
- Rule: pydantic-ai 1.x does NOT have a one-shot `OpenAIModel(base_url=..., api_key=...)` constructor. Model and provider are separated: `OpenAIChatModel("<model-name>", provider=OpenAIProvider(base_url="...", api_key="..."))`. Import from `pydantic_ai.models.openai` and `pydantic_ai.providers.openai` respectively. This is a rename from the 0.x API.
- Evidence: source reads + spike round-trip at `scratchpad/spikes/pydanticai_full_stack_spike.py` Part A (routed to LiteLLM proxy at localhost:4000).
- Related docs: §0 C8 in the integration brainstorm; P1.2 in the roadmap.

### 2026-04-10 — Temporal 2 MB payload limit binds in Bodha for unbounded-domain background agents; cross the boundary by reference

- Scope: background agent input contracts for any agent whose input scales with session length, entity memory count, or community size.
- Trigger: Codex critique 2026-04-10 of §0 C7 — pointing to specific Bodha job-plan sections with genuinely unbounded inputs.
- Rule: Bodha background agents with unbounded-domain inputs (F1.1–F1.3 session-end consolidation, E2.2 entity profile synthesis, E1.5a community consolidation, any future "aggregate everything in scope X" job) MUST cross the Temporal boundary by ID/reference. The input contract carries `community_id + member_memory_ids`, not `member_memories: list[Memory]`. The agent's first `@agent.tool` inside the activity fetches actual data via `deps.postgres.load_memories(ids)`. This keeps payloads bounded regardless of scope growth. Earlier estimate "background agents carry per-call payloads, not session history" was wrong — it held for single-turn extraction agents but not for aggregation agents.
- Evidence: `bodha-job-plan-v2_7.md` §§5.1, 5.2, 6.1 explicitly enumerate the unbounded inputs. Codex citations at those sections.
- Related docs: §0.1 C7-DEFECT-fix in the integration brainstorm; P2.1, P3.1, P6.2b in the roadmap.

### 2026-04-10 — TemporalAgent wiring pattern: PydanticAIPlugin + PydanticAIWorkflow + __pydantic_ai_agents__

- Scope: running a PydanticAI agent as a Temporal workflow with proper activity registration, sandbox passthrough, and data converter setup.
- Trigger: writing the P1.0 hard-blocker spike to prove `dataclasses.replace` mutation survives the activity boundary.
- Rule: The minimal wiring for a TemporalAgent-as-workflow is:
  1. Build `Agent(...)` and `TEMPORAL_AGENT = TemporalAgent(agent, name='...')` at module level (not inside the workflow class, because the workflow sandbox restricts imports and I/O).
  2. Define `class MyWorkflow(PydanticAIWorkflow)` with `__pydantic_ai_agents__: Sequence[TemporalAgent[Any, Any]] = [TEMPORAL_AGENT]` as a class attribute. Decorate the class with `@workflow.defn(name='...')`. The workflow's `@workflow.run` async method calls `await TEMPORAL_AGENT.run(prompt, deps=...)`.
  3. Connect the client with the plugin: `Client.connect(host, plugins=[PydanticAIPlugin()])`. The plugin auto-installs the Pydantic data converter and the workflow sandbox passthrough for `pydantic_ai`, `pydantic`, `httpx`, etc.
  4. Start a worker: `Worker(client, task_queue=..., workflows=[MyWorkflow])`. The plugin's `configure_worker` hook walks `__pydantic_ai_agents__` on every registered workflow class and auto-extracts `agent.temporal_activities` into the worker's activity list — no manual activity registration needed.
  5. Submit via `client.execute_workflow(MyWorkflow.run, prompt, id=..., task_queue=...)`.
- Evidence: end-to-end proven by `scratchpad/spikes/repack_mutation_temporal_spike.py` on 2026-04-10. Sentinel swap "What is the capital of France?" → "What is the capital of Germany?" via `before_model_request` produced `"Berlin"` when submitted through the workflow. Activity boundary round-trip took ~3 seconds wall clock against local `bodha-temporal` container.
- Related docs: `.venv/lib/python3.12/site-packages/pydantic_ai/durable_exec/temporal/__init__.py:94-128` (PydanticAIPlugin), `_workflow.py` (PydanticAIWorkflow base class), §0.1 "P1.0" in the integration brainstorm.

### 2026-04-10 — before_model_request mutation via dataclasses.replace survives the TemporalAgent activity boundary

- Scope: the C1 correction — ContextRepackCapability, ECBCapability, ObservabilityCapability, and any other capability that mutates `ModelRequestContext` in `before_model_request` or `after_model_request` or `wrap_model_request`.
- Trigger: Codex critique flagged that the plain-`Agent.run` proof was not sufficient; the same mutation must survive the workflow↔activity boundary for the production plan to hold.
- Rule: `dataclasses.replace(request_context, messages=...)` mutations from `before_model_request` reach the actual LLM request body even when the agent is wrapped in `TemporalAgent`. No special handling needed on either side of the activity boundary. The mutated context is serialized into `_RequestParams` inside the activity, used to build the HTTP request, and the response propagates back through the activity boundary unchanged. This confirms the C1 correction is sound end-to-end.
- Evidence: P1.0 spike at `scratchpad/spikes/repack_mutation_temporal_spike.py` on 2026-04-10 — workflow submitted France-prompt, capability swapped to Germany-prompt inside the activity, model answered "Berlin", workflow returned "Berlin" to the driver. Full round-trip through a real Temporal workflow against the local `bodha-temporal` container.
- Related docs: §0.1 C1-fix and C1-proof in the integration brainstorm; P1.0 (now DONE) and P5.5/P5.6/P5.7 in the roadmap.

### 2026-04-10 — prefer empirical spike over doc-level reasoning when stakes are architectural

- Scope: verifying any API assumption that will be load-bearing for multiple downstream phases.
- Trigger: C1 mutation path was "obvious" from the type signature, but running the spike caught that `ModelRequestContext.model_copy()` doesn't exist — a runtime crash that would have surfaced deep inside P5 instead of pre-P1.
- Rule: When a design decision rests on "the library surface says X will work," spend 30 minutes writing a minimal spike that demonstrates X actually works end-to-end before committing to the design. Doc verification and source reading are necessary but not sufficient. Empirical evidence has caught two wrong assumptions so far in this integration (model_copy, C7 payload assumption) that neither Context7 docs nor Codex review caught directly.
- Evidence: `scratchpad/spikes/repack_mutation_test.py` first run crashed at `model_copy`; second run proved the correct path. This was a 30-minute investment that prevented hours of P5 debugging.
- Related docs: §0.1 all of it in the integration brainstorm. Also AGENTS.md "Library Docs Lookup" / docs-researcher pattern.
