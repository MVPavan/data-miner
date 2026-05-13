---
name: bodha-memory-eval
description: >
  Version-agnostic stress-testing of Bodha design documents against memory evaluation datasets.
  Given any design version (v_2_5, v_2_6, v_3_0, etc.) and any eval dataset version/document
  (DS-15, DS-20, DS-30, DS-40, DS-50, DS-80, or future), dynamically discovers design docs,
  generates a version-specific evaluation checklist, and produces a structured evaluation report.
  Use this skill whenever the user mentions "memory eval", "memory test", "stress test design",
  "evaluate design against DS-", "run eval on v2.5/v2.6/v3.0", "validate design version",
  "test design against eval", "run DS-15/DS-20/DS-30/DS-40/DS-50/DS-80", "trace scenarios",
  "check design coverage", "run bodha eval", "run memory scenarios", or any request to validate
  Bodha design documents against evaluation datasets — even if they don't use the word "eval".
  Also trigger when the user references specific eval dataset files or scenario IDs (e.g., DS20-16).
---

# Bodha Memory Eval — Version-Agnostic Design Validation

Validate any version of Bodha's design documents against any version of the memory evaluation
datasets. Unlike a hardcoded test, this skill dynamically discovers design docs, builds a
version-specific checklist, and traces each eval scenario through the design's architecture.

The eval datasets are version-independent behavioral contracts — they describe conversations,
questions, and expected outputs without referencing any architecture. This skill bridges that gap:
it takes those behavioral expectations and validates whether a given design version can deliver.

---

## Input Parsing

Extract three parameters from the user's request:

| Parameter | Example | Maps To |
|-----------|---------|---------|
| **Design version** | `v_2_5`, `v_2_6` | `docs/design/<version>/core/` |
| **Eval version** | `v_1`, `v_2` | `docs/design/memory-design-tests/<version>/` |
| **Eval document(s)** | `ds15`, `ds40`, `all` | `bodha-eval-<doc>.md` within eval version |

If the user says "run DS-20 on v2.5 core", that means: eval doc = `ds20`, design version = `v_2_5`.

If any parameter is ambiguous or missing, ask the user. If the eval version isn't specified,
default to the latest available (check the directory). Confirm the resolved parameters before
proceeding.

The user may also specify:
- A scenario range: "DS20-16 through DS20-20" — process only those scenarios
- Multiple documents: "ds15 and ds20" — process both sequentially
- Include discussions: "include discussions too" — also read `new_discussions/` directory

---

## Phase 1: Discovery

Scan the design version's directory to catalog what you're validating. This builds the foundation
for everything that follows.

### Step 1.1 — Catalog design documents

List all `.md` files in `docs/design/<design_version>/core/`. For each, record filename and size.

Spawn **parallel subagents** (one per design document, use Opus — this is deep analysis) to read
each file and produce a structured summary:

```
FILE: <filename> (<size>)
COMPONENT: <component name inferred from filename>
AUTHORITY OVER: <1-2 line scope description>
KEY SECTIONS:
  - §<heading> (line N): <what it covers>
  - §<heading> (line N): <what it covers>
PIPELINE ROLE: <hot path / cold path / async / infrastructure / cross-cutting>
KEY CONTRACTS: <data structures, interfaces defined — name and purpose>
INVARIANTS: <numbered invariants with their text, e.g., "INV-12: Postgres is sole write authority">
MEMORY CHECK RELEVANCE: <which of the 13 check types this component is most relevant to>
STORES DEFINED/USED:
  - <store name> (<technology>): <role — authority/projection/cache>
    Schemas: <tables, collections, graphs with key fields>
    Written by: <which components/pipeline stages>
    Read by: <which components/pipeline stages>
    Key fields for eval: <temporal fields, entity refs, semantic keys, validity flags>
```

If the document is a database/storage-focused component (like a "Chitta" equivalent), extract the
full schema details — table names, column names, index strategies, relationships between tables.
This level of detail matters because eval scenarios often hinge on whether the storage model can
represent the required information (temporal bounds, entity relationships, memory types).

### Step 1.2 — Build pipeline model

From the component summaries, reconstruct the processing pipeline for this design version:

- **Hot path:** User message → [which components, in what order] → Response
- **Cold path:** [Extraction trigger] → [evaluation] → [storage projections]
- **Async path:** [Background jobs, consolidation, enrichment]
- **Store interactions:** Which stores exist and what each is responsible for

This is derived fresh from whatever design version you're reading — never assume v2.5's pipeline.

---

## Phase 2: Checklist Generation

The checklist is the key intermediary artifact. It translates abstract design documents into a
concrete evaluation plan tailored to this specific design version. It's generated once per design
version and reused across eval runs.

### Check for existing checklist

Look for: `docs/design/memory-design-tests/<eval_version>/checklists/checklist-<design_version>.md`

If it exists and the user hasn't asked to regenerate → read it and skip to Phase 3.

### Step 2.1 — Read check type definitions

Read the check type reference table from any eval dataset in the eval version directory. This gives
you the memory check types (SFR, MHR, TRA, KUC, AHA, IPE, DEL, EEM, RSM, PCT, NDM, MGV, DIM)
with their definitions.

Also extract the cross-cutting tags (e.g., `multi_party_attribution`, `observation_memory`,
`multi_source`, `implicit_feedback_learning`, `motivation_inference`) and their meanings.

### Step 2.2 — Map check types to design sections

Using the component summaries from Phase 1, build a mapping for each check type:
- Which design documents and sections are relevant
- Which pipeline stages are exercised
- Which invariants apply
- What key questions to ask when evaluating scenarios of this type

Do the same for cross-cutting tags.

### Step 2.3 — Write the checklist

Save to `docs/design/memory-design-tests/<eval_version>/checklists/checklist-<design_version>.md`:

```markdown
# Evaluation Checklist — Bodha <design_version>

**Generated:** <timestamp>
**Design version path:** docs/design/<design_version>/core/
**Design documents:** <count> core files

## Design Documents Catalog

| Document | Component | Authority Over |
|----------|-----------|---------------|
| <filename> | <component> | <scope> |

## Pipeline Model

### Hot Path
<ordered pipeline steps with component ownership — derived from Phase 1>

### Cold Path
<extraction → evaluation → storage pipeline>

### Async Path
<background jobs, consolidation, enrichment>

### Store Architecture

For each persistent store discovered in the design docs, document:

| Store | Technology | Authority / Role | Key Schemas/Collections | Read By | Written By |
|-------|-----------|-----------------|------------------------|---------|------------|
| <name> | <tech> | <what it's authoritative for> | <tables, collections, or graphs> | <components> | <components> |

Also note:
- **Write authority model:** Which store is the source of truth? Are others projections/caches?
- **Consistency guarantees:** How are stores kept in sync? (e.g., outbox pattern, eventual consistency)
- **Key data structures:** For each store, list the primary schemas/tables/collections with their
  key fields (especially fields relevant to memory evaluation: temporal fields, entity references,
  semantic keys, memory types, validity flags)
- **Store interaction per pipeline stage:** Which stores are read/written at each pipeline step

## Check Type → Design Section Map

### SFR — Single-Hop Factual Recall
- **Primary sections:** <doc §section>, <doc §section>
- **Pipeline stages:** <which stages execute for this check type>
- **Invariants:** <relevant invariant numbers>
- **Key questions:** <what to verify when evaluating SFR scenarios>

[... repeat for each check type ...]

## Cross-Cutting Tag Map

### multi_party_attribution
- **Relevant sections:** <doc §section>
- **Key mechanism:** <how the design handles multi-party attribution>

[... repeat for each tag ...]

## Invariants Reference
<numbered list of all invariants from design docs, grouped by source file>
```

---

## Phase 3: Scenario Evaluation

This is the core analytical work. Each scenario gets a deep, independent evaluation.

### Step 3.1 — Parse the eval dataset

Read the specified eval document(s). Parse:
- All scenario IDs, check types, difficulty, session span, evidence count, eval method, tags
- Which scenarios have full content vs are inherited references (DS-20/40/80 inherit from predecessors)
- Apply any user-specified subset filter

For inherited scenarios without content in the file, note them as skipped — never fabricate content.

### Step 3.2 — Create the report file

Create the report at:
`docs/design/memory-design-tests/<eval_version>/results/<design_version>-<eval_doc>-results.md`

Initialize with the header template from `references/output-format.md`.

### Step 3.3 — Evaluate scenarios using subagents

For each scenario with full content, spawn a subagent (use Opus — design validation demands deep
reasoning). Each subagent receives:

1. The full scenario content (input sessions, question, expected output, evidence keys, failure modes)
2. The checklist's check type mapping for this scenario's type and tags
3. The design version path so it can read actual design doc sections

**Subagent prompt pattern:**

```
You are evaluating a memory system design against a behavioral scenario.

SCENARIO:
<full scenario content>

CHECKLIST GUIDANCE (for check type <CODE>):
<relevant section from checklist>

DESIGN DOCS PATH: docs/design/<design_version>/core/

TASK:
1. Read the design doc sections listed in the checklist guidance.
   Read the ACTUAL files — the checklist is a pointer, not a substitute.
2. Trace this scenario through the design's pipeline:
   - Hot path: classification/routing → entity resolution → retrieval → reranking
   - Cold path: extraction → evaluation verdicts → store projections
   - Async path: relevant background jobs and consolidation effects
3. For each observation, classify as:
   DESIGN HANDLES (cite section) | DESIGN GAP | PARTIAL COVERAGE | INVARIANT TENSION | BENCHMARK INSIGHT
   With severity: Critical / High / Medium / Low
4. For each Common Failure Mode listed, identify whether the design prevents it.
5. Return findings in this exact format:
   <output format from references/output-format.md>
```

**Parallelism:** Launch subagents in parallel batches. For large datasets (DS-50, DS-80), batch
3-5 scenarios at a time to balance throughput with resource limits.

### Step 3.4 — Compile results

As each subagent completes, append its structured output to the report file. Update the scenario
index table (status → done, fill findings count) and the aggregate metrics.

---

## Phase 4: Cross-Scenario Synthesis

After all scenarios are evaluated, step back and analyze patterns. This is where systemic issues
become visible — individual scenario findings may look like edge cases, but repeated patterns
reveal architectural gaps.

Append to the report:

### Design Strengths
Capabilities the design handles well across multiple scenarios, with evidence.

### Systemic Gaps
Gaps appearing across 2+ scenarios — these are architectural issues, not edge cases.

### Invariant Tensions
Cases where design invariants conflict or create unexpected constraints.

### Check Type Coverage Assessment

| Check Type | Code | Scenarios Tested | Design Coverage | Notes |
|-----------|------|-----------------|-----------------|-------|
| [name] | [code] | [scenario IDs] | Strong / Partial / Weak | [brief explanation] |

### Top Critical Findings
Ranked by severity × frequency across scenarios.

### Recommendations
Prioritized design improvements, referencing specific invariants, contracts, or sections.

Update the report status to `**Status:** Complete` and present the report path to the user with
a brief verbal summary of top findings.

---

## Subagent Strategy

Design validation is precision work that benefits from deep reasoning. Use the largest available
model (Opus) for all subagents.

| Subagent | Phase | Count | Parallel? |
|----------|-------|-------|-----------|
| Discovery Reader | 1.1 | 1 per design doc | Yes, all parallel |
| Checklist Builder | 2.2 | 1 total | No (needs all summaries) |
| Scenario Evaluator | 3.3 | 1 per scenario | Yes, batches of 3-5 |
| Synthesizer | 4 | 1 total | No (needs all findings) |

**Discovery Readers** are lightweight — they summarize structure and contracts.
**Scenario Evaluators** are heavyweight — they read design docs, trace pipelines, and reason deeply.

---

## Important Rules

1. **Read before you cite.** Every claim about the design must come from reading the actual file,
   not from the checklist alone. The checklist is a map — the design docs are the territory.

2. **Version-agnostic always.** Never hardcode paths, filenames, or version numbers. Derive
   everything from the user's specified version. If the design version has 7 core docs instead of
   9, or uses different component names, adapt.

3. **Reuse checklists.** If a checklist exists for this design version, use it unless the user
   explicitly asks to regenerate. Checklists are expensive to create but stable once made.

4. **Core only by default.** Validate against `core/` documents only. Include `new_discussions/`
   only if the user explicitly asks.

5. **Don't fabricate scenarios.** DS-20/40/80 files reference predecessor scenarios without
   including their content. Skip these with a note — never invent scenario content.

6. **Tags are additional checks.** Tags like `multi_party_attribution` point to sub-capabilities
   that need verification beyond the main check type. Don't ignore them.

7. **Check invariants actively.** Design documents define numbered invariants (system guarantees).
   When tracing a scenario, check whether relevant invariants hold. Violations are high-severity.

8. **Handle subsets gracefully.** "Test DS80-51 through DS80-55" means process only those five.

9. **Benchmarks if available.** If the design version includes a benchmarks document, cross-reference
   scenario findings against published benchmark data. If no benchmarks doc exists, skip this.

10. **Report format matters.** Follow the output format in `references/output-format.md` precisely.
    Consistent formatting makes reports comparable across design versions and eval runs.

---

## Optional: Cross-Version Comparison

Only when the user explicitly asks to compare two design versions (e.g., "compare v2.5 and v2.6
against DS-15"), run the full evaluation for each version independently, then produce a comparison
report.

### Comparison workflow

1. Run Phase 1–4 for design version A → produces report A
2. Run Phase 1–4 for design version B → produces report B (checklists may already exist)
3. Produce a comparison report at:
   `docs/design/memory-design-tests/<eval_version>/results/comparison-<vA>-vs-<vB>-<eval_doc>.md`

### Comparison report structure

```markdown
# Design Version Comparison: <vA> vs <vB>

**Eval Dataset:** <filename>
**Eval Version:** <eval_version>

## Summary

| Metric | <vA> | <vB> | Delta |
|--------|------|------|-------|
| DESIGN HANDLES | X | Y | +/- |
| DESIGN GAP | X | Y | +/- |
| PARTIAL COVERAGE | X | Y | +/- |
| INVARIANT TENSION | X | Y | +/- |
| Critical findings | X | Y | +/- |

## Per-Scenario Comparison

### <scenario_id>: <title>
- **<vA>:** <brief summary of findings>
- **<vB>:** <brief summary of findings>
- **Change:** <what improved, regressed, or stayed the same>

## Regressions
<Scenarios where vB is worse than vA — these need attention>

## Improvements
<Scenarios where vB is better than vA>

## Recommendations
<What to address before adopting vB>
```

Do not build this comparison unless the user requests it.
