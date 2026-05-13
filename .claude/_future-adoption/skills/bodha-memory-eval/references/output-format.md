# Output Format Reference — Bodha Memory Eval

This document defines the exact report structure that scenario evaluators must produce.
Consistent formatting makes reports comparable across design versions and eval runs.

---

## Report Header

```markdown
# Bodha Design Evaluation Report

**Design Version:** <version> (docs/design/<version>/core/)
**Eval Dataset:** <filename>
**Eval Version:** <eval_version>
**Scenarios:** <count processable> / <count total> (inherited without content: <count skipped>)
**Checklist:** <checklist path>
**Started:** <timestamp>
**Status:** In Progress
```

---

## Scenario Index Table

```markdown
## Scenario Index

| # | ID | Check Type | Difficulty | Status | Findings | Critical |
|---|-----|-----------|------------|--------|----------|----------|
| 1 | DS20-16 | MHR | Medium | ⏳ | — | — |
```

Update each row as scenarios complete: status → ✅, fill findings count and critical count.

---

## Aggregate Metrics

```markdown
## Aggregate Metrics

| Metric | Count |
|--------|-------|
| Scenarios completed | 0 / <total processable> |
| DESIGN HANDLES | 0 |
| DESIGN GAP | 0 |
| PARTIAL COVERAGE | 0 |
| INVARIANT TENSION | 0 |
| BENCHMARK INSIGHT | 0 |
| Critical | 0 |
| High | 0 |
| Medium | 0 |
| Low | 0 |
```

Update counts after each scenario completes.

---

## Per-Scenario Section

Each scenario evaluator subagent must return output in this exact structure:

```markdown
---

## <ID>: <Title>

**Check Type:** <code> — <full name> | **Difficulty:** <diff> | **Tags:** <tags>
**Benchmark Coverage:** <which published benchmarks test this, if benchmarks doc available>

### Pipeline Trace

**Hot Path:**
- Classification/Routing: <what template/routing and parameters>
- Entity Resolution: <resolution mechanism and result>
- Retrieval Stages: <stage1 → stage2 → ...>
- Reranking: <boosts/penalties applied>
- Feedback Loop: <would it trigger? why/why not>

**Cold Path:**
- Extraction Candidates: <list with memory_type, semantic_key, entities>
- Evaluation Verdicts: <PASS/HOLD/REJECT per candidate with reasoning>
- Store Projections: <what goes to which store>

**Async Path:**
- Background Jobs: <relevant jobs and expected effects>
- Consolidation: <what would be refined/merged>

### Findings

| # | Type | Severity | Description | Design Doc Evidence |
|---|------|----------|-------------|---------------------|
| 1 | DESIGN HANDLES | — | <description> | <component-doc §section> |
| 2 | DESIGN GAP | High | <description> | Not found in any design doc |
| 3 | PARTIAL COVERAGE | Medium | <description> | <component-doc §section> (general case only) |
| 4 | INVARIANT TENSION | High | <description> | INV-<N> vs INV-<M> |
| 5 | BENCHMARK INSIGHT | — | <description> | <benchmark-name>: <relevant data> |

### Failure Mode Analysis

| Failure Mode (from dataset) | Design Prevention Mechanism | Verdict |
|----------------------------|---------------------------|---------|
| <mode from scenario> | <mechanism from design, or "none found"> | ✅ Prevented / ⚠️ Partial / ❌ Unaddressed |

### Benchmark Context

<Published benchmark data relevant to this check type, if benchmarks doc available.
If no benchmarks doc in this design version, write "No benchmarks document in this design version.">
```

---

## Finding Types Reference

| Type | Meaning | When to Use |
|------|---------|-------------|
| **DESIGN HANDLES** | Design explicitly covers this requirement | You can cite a specific section that addresses the scenario's need |
| **DESIGN GAP** | Design doesn't address this need | You searched relevant docs and found no mechanism |
| **PARTIAL COVERAGE** | Design covers the general case but not this edge | The mechanism exists but doesn't handle the specific scenario variant |
| **INVARIANT TENSION** | Scenario reveals conflict between design guarantees | Two invariants or principles pull in opposite directions |
| **BENCHMARK INSIGHT** | Published benchmark has relevant data | A known benchmark tests this same capability with measurable results |

## Severity Levels

| Severity | Meaning |
|----------|---------|
| **Critical** | System would produce wrong results, lose data, or fail silently |
| **High** | Key capability is unspecified or would silently degrade |
| **Medium** | Edge case with a workaround, or affects non-critical paths |
| **Low** | Documentation gap, cosmetic issue, or theoretical concern |

DESIGN HANDLES findings typically have no severity (use "—") since they indicate success.
BENCHMARK INSIGHT findings also typically have no severity — they provide context, not problems.

---

## Cross-Scenario Synthesis Section

Appended after all scenarios are evaluated:

```markdown
---

## Cross-Scenario Synthesis

### Design Strengths
<Capabilities well-handled across multiple scenarios, with scenario IDs as evidence>

### Systemic Gaps
<Gaps appearing in 2+ scenarios — architectural issues, not edge cases.
Group by theme, list affected scenario IDs>

### Invariant Tensions
<Cases where design invariants conflict. Cite the specific invariants and scenarios>

### Check Type Coverage Assessment

| Check Type | Code | Scenarios Tested | Design Coverage | Notes |
|-----------|------|-----------------|-----------------|-------|
| Single-Hop Factual Recall | SFR | <IDs> | Strong / Partial / Weak | <brief> |
| Multi-Hop Cross-Session | MHR | <IDs> | Strong / Partial / Weak | <brief> |
| Temporal Reasoning | TRA | <IDs> | Strong / Partial / Weak | <brief> |
| Knowledge Update | KUC | <IDs> | Strong / Partial / Weak | <brief> |
| Abstention | AHA | <IDs> | Strong / Partial / Weak | <brief> |
| Implicit Preference | IPE | <IDs> | Strong / Partial / Weak | <brief> |
| Dense Extraction | DEL | <IDs> | Strong / Partial / Weak | <brief> |
| Episodic Memory | EEM | <IDs> | Strong / Partial / Weak | <brief> |
| Reflective Memory | RSM | <IDs> | Strong / Partial / Weak | <brief> |
| Persona Consistency | PCT | <IDs> | Strong / Partial / Weak | <brief> |
| Non-Declarative | NDM | <IDs> | Strong / Partial / Weak | <brief> |
| Memory Governance | MGV | <IDs> | Strong / Partial / Weak | <brief> |
| Durable Instruction | DIM | <IDs> | Strong / Partial / Weak | <brief> |

### Top 5 Critical Findings
<Ranked by severity × frequency across scenarios.
Each entry: finding description, affected scenarios, design doc reference, recommendation>

### Recommendations
<Prioritized design improvements.
Each: what to change, which design doc(s), which invariants affected, expected impact>
```

Update report header: `**Status:** Complete`
