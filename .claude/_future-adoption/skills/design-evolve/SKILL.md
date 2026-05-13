---
name: design-evolve
description: >
  Evolve versioned design documents by integrating new discussion files into core specs
  to produce a self-contained next version. Use this skill whenever the user wants to
  create a new version of design documents by merging discussions, decisions, or new specs
  into the existing core design docs. Triggers on phrases like "integrate discussions",
  "new design version", "merge decisions into core", "evolve the docs", "create v2.6 from v2.5",
  or any request to update design specifications based on recent discussions, decision logs,
  or scratchpads. Also use when the user says "upgrade the design", "consolidate discussions
  into core", or references a folder structure with core/ and discussions/ subfolders.
---

# Design Evolve — Versioned Design Document Integration

You are integrating new discussion files into an existing set of core design documents to produce a new, self-contained version. The new version is the single source of truth — it must not reference older versions internally, and must contain everything needed to understand the design.

This is precision work. The documents you are evolving may be the architectural foundation for a complex system. A hallucinated detail, a lost invariant, or a misinterpreted decision can cause real damage downstream. Treat every sentence with the care of an engineer reviewing a spec before implementation begins.

---

## Principles

**Zero hallucination.** Every sentence in the output must trace to either the existing core or a discussion file. Never invent information, fill gaps with assumptions, or "improve" the design beyond what the sources say. If something is unclear, ask the user.

**Self-contained output.** The new version's documents are a complete unit. No "see v2.5 for details" or "as discussed in the old discussions." Every piece of information that matters is present in the new version. The only place where referencing prior versions is appropriate is in a version history table (changelog).

**Incremental context.** Never load all files at once. Process one concept at a time. Use sub-agents to read and analyze files so the main context stays clean and focused. This is not optional — design docs can be 20-50KB each, and loading them all degrades accuracy.

**Source fidelity.** Preserve original intent, terminology, and technical precision. If the source says "5-minute TTL," write "5-minute TTL" — not "short TTL." If it says "Qdrant," write "Qdrant" — not "vector database." Maintain exact technical terms and values.

**User checkpoints.** When you encounter ambiguity, vagueness, contradictions between sources, or excessive repetition, stop and ask the user before proceeding. Do not guess. The cost of a 30-second question is far less than the cost of a wrong integration.

---

## Workflow

### Phase 0: Setup

Gather from the user (or extract from their message if already provided):

1. **Source version path** — directory containing `core/` and a discussions folder (e.g., `docs/design/v_2_5/`)
2. **Target version identifier** — the version number for the output (e.g., "2_6", "3_0")
3. **Target path** — where to create the new version (default: sibling of source following the same naming convention, e.g., `docs/design/v_2_6/`)

The target directory should mirror the source structure. The output goes into `core/` inside the target:

```
<target_path>/
└── core/
    ├── <updated-doc-1-with-new-version>.md
    ├── <updated-doc-2-with-new-version>.md
    └── ...
```

Confirm these with the user, then create the target directory. When renaming files for the new version, follow the same naming convention as the source (e.g., if source files use `*-v2_5.md`, target files should use `*-v2_6.md`).

---

### Phase 1: Discovery & Concept Mapping

**Goal:** Understand all discussion files, group them by concept, and plan the integration order.

#### Step 1.1 — Inventory

List all files in both `core/` and the discussions folder. Record names and sizes. This gives you the landscape.

#### Step 1.2 — Scan each discussion file

Spawn a sub-agent **per discussion file** (in parallel — they are independent reads). Each sub-agent should:

1. Read the entire file
2. Produce a structured report:

```
FILE: <filename> (<size>)
SUMMARY: <3-5 sentences — what this file is about>

KEY DECISIONS/CHANGES:
- <Decision 1> — affects <core-doc> §<section>
- <Decision 2> — affects <core-doc> §<section>
...

NEW CONCEPTS: <any entirely new subsystems or components introduced>
CROSS-REFS: <references to other discussion files>
MATURITY: <final decision | working notes/scratchpad | comparison analysis | integration guide>
```

The MATURITY field matters — scratchpads and brainstorms need user confirmation before integration, while final decision documents can proceed more confidently.

#### Step 1.3 — Group by concept

Analyze the reports and group discussion files that address the same subsystem or concept. A "concept" is a coherent cluster of changes that should be integrated together because they touch the same parts of the core.

Rules for grouping:
- Files that reference each other likely belong to the same concept
- A decision log + an integration guide + a discussion log about the same topic = one concept
- A file may span multiple concepts — note which parts belong where
- Comparison/analysis documents (e.g., "X vs Y") belong to the concept of whatever was decided

#### Step 1.4 — Present integration plan to user

Show the user:

1. **Concept groups** — which discussion files belong to each, and a 1-line summary per concept
2. **Proposed order** — process self-contained concepts first; cross-cutting changes last
3. **Flags** — any concerns: vague content, contradictions between files, scratchpads that may not be ready, heavy repetition across files
4. **Affected core docs per concept** — which core files each concept will modify

**Wait for user approval before proceeding.** If they want to adjust grouping, reorder, or exclude certain discussions, incorporate that.

---

### Phase 2: Concept-by-Concept Integration

This is the core loop. For each concept group, in the agreed order:

#### Step 2.1 — Deep-read discussions

Use a sub-agent to read ALL discussion files for this concept (full content, not summaries). The sub-agent produces:

```
CONCEPT: <name>
FILES READ: <list>

DECISIONS & CHANGES:
1. <What changed> — Source: <file> §<section>
   Details: <precise description preserving technical specifics>
   Affects: <core-doc> §<section>

2. ...

NEW SECTIONS NEEDED:
- <If discussions introduce new subsystems needing new core sections or files>

OPEN QUESTIONS:
- <Anything unclear, contradictory, or under-specified>

REPETITION:
- <Same point made in multiple files — list all occurrences so we can consolidate>
```

#### Step 2.2 — Deep-read affected core sections

Use a sub-agent to read the core documents (or relevant sections) that this concept affects. The sub-agent notes:

- Current content of each affected section
- Structural conventions (heading hierarchy, table formats, invariant numbering, naming patterns)
- Cross-references to other core documents that may need updating
- Any existing content that the discussions will supersede

#### Step 2.3 — Build the integration checklist

From Steps 2.1 and 2.2, create a precise, numbered checklist:

```
CONCEPT: <name>

INTEGRATION CHECKLIST:
□ 1. <core-doc> §<section>: <specific change>
     Source: <discussion-file> §<section>
     Action: update text | add subsection | replace section | add table row | update invariant

□ 2. <core-doc> §<section>: <specific change>
     Source: <discussion-file> §<section>
     Action: ...

...

□ N. Create new file <filename> for <topic> (if needed)
     Source: <discussion-files>

NEEDS USER INPUT:
- <Any ambiguity, contradiction, or judgment call>
```

**If there are items needing user input, present the checklist and WAIT.** Do not proceed until the user resolves these.

If the checklist is clean, present it briefly (the user should see what you plan to do) and proceed.

#### Step 2.4 — Execute integration

For each checklist item:

1. If this core file hasn't been created in the target yet, **copy the ENTIRE source core file** as the starting point. Never write a partial file.
2. Read the current content of the target section.
3. Apply the change.
4. Write the updated content to the target file.

**Rules during execution:**

- **Version headers:** Update version number, date, and status in every file header.
- **Version notes that describe deltas:** If the source says "v2.5 adds X" or "New in v2.5: Y" — rewrite so X and Y are described as current facts, not as additions. The version history table is the one exception where referencing prior versions is correct.
- **Invariants and design decisions:** Preserve all unless a discussion explicitly supersedes one. When superseding, use the same numbering/naming style and note the new decision.
- **Cross-references:** Update filenames in cross-references to the new version (e.g., `bodha-chitta-v2_5.md` → `bodha-chitta-v2_6.md`).
- **Tables:** When adding rows to tables, match the existing column structure and style exactly.
- **New sections:** Place them where they logically belong in the document's existing structure, not appended at the end. Match the heading level and style.

#### Step 2.5 — Verify this concept

After completing all checklist items:

1. Re-read the modified sections in the target files
2. Confirm every change traces to a source (discussion file or existing core)
3. Confirm no old version references leaked in
4. Confirm the document reads coherently around the changes

Mark the concept as done. Move to the next one.

---

### Phase 3: Cross-Document Consistency

After all concepts are integrated:

#### Step 3.1 — Copy untouched files

Any core files NOT modified by any concept must still be copied to the new version — with updated version numbers, dates, filenames, and version-reference cleanup. They are part of the complete set.

#### Step 3.2 — Version reference sweep

**Use grep/search tools** to find ALL occurrences of the old version identifier (version numbers, old filenames) across the new core files. Update every one. Do not rely on memory — use the tools.

Search patterns to check:
- The old version number in any format (e.g., "v2.5", "v2_5", "2.5", "version 2.5")
- Old filenames (e.g., `*-v2_5.md`)
- Phrases like "new in v2.5", "v2.5 adds", "added in v2.5"

#### Step 3.3 — Cross-reference audit

Verify all cross-references between core documents point to correct new-version filenames and section numbers. Sections may have been renumbered or renamed during integration.

#### Step 3.4 — Repetition consolidation

If you noticed the same concept explained substantively in multiple core documents during integration, flag it to the user:
- Where each instance appears
- Whether to consolidate to one location with cross-references, or keep both (if they serve different contexts)

**Ask the user** — don't decide this unilaterally.

#### Step 3.5 — Version history

In the main design document, update the version history table to include the new version. Describe what it adds, using the same style as prior entries. This is the ONE place where referencing previous versions is appropriate.

---

### Phase 4: Summary & Handoff

Present the user with:

1. **Integration summary** — what was integrated, concept by concept, in 2-3 sentences each
2. **New/modified files** — full list of files in the new `core/`
3. **Key decisions** — integration decisions you made (where content landed, how conflicts were resolved)
4. **Deferred items** — anything from discussions that was too vague, out of scope, or explicitly excluded
5. **Suggested follow-ups** — any work implied by the discussions that wasn't in scope for this integration

---

## Anti-Hallucination Protocol

These rules are non-negotiable. They exist because document integration is high-stakes precision work where a fabricated detail can propagate into implementation.

1. **Read before write.** Never write content without having the source material in context (directly or via a sub-agent report). "I remember it said something about caching" is not good enough — read the actual text.

2. **No gap-filling.** If a discussion says "add caching" but doesn't specify the strategy, write exactly that ("caching will be added; strategy TBD") or ask the user. Do not invent a caching strategy.

3. **Preserve precision.** Keep exact numbers, names, identifiers, and technical terms. `pplx-embed-v1` stays as `pplx-embed-v1`. `43,200 workflow executions/day` stays as `43,200 workflow executions/day`.

4. **Quote when uncertain.** If you're unsure how to rephrase without losing meaning, keep the original wording. Faithful reproduction beats elegant paraphrase.

5. **Track provenance internally.** While working, know why you wrote each thing. "I wrote X because discussion file Y §Z says W." You don't need to output this, but you need it for verification.

6. **Flag contradictions, don't resolve them.** If two sources disagree, stop and ask the user. Don't pick the one that "seems newer" or "makes more sense."

7. **Don't improve the design.** Your job is integration, not design review. If you see something that seems wrong or suboptimal in the source material, integrate it faithfully and optionally mention your concern in the Phase 4 summary. Do not silently "fix" it.

---

## Sub-Agent Patterns

Sub-agents are essential for this work. They keep the main context clean and allow parallel processing of large files.

**Discussion Scanner** (Phase 1.2) — one per discussion file, all in parallel:
```
Read <file> thoroughly. Produce a structured report with:
summary, key decisions/changes, affected core docs, new concepts,
cross-references, maturity level (final decision vs scratchpad vs analysis).
```

**Core Reader** (Phase 2.2) — one per affected core document:
```
Read <core-file>. Extract content of sections <list>.
Note structural conventions (heading levels, table formats, numbering).
Note cross-references to other core documents.
```

**Concept Integrator** (Phase 2.4, for large concepts with >3 files):
```
Given: discussion summaries, affected core sections, integration checklist.
Execute each checklist item: read target section, apply change, write result.
Return: the updated file contents.
```

**Consistency Checker** (Phase 3):
```
Search all files in <target_path>/core/ for:
- Old version references (grep for patterns)
- Broken cross-references
- Duplicate substantial content across files
Return: list of issues with file:line locations.
```

**When not to use sub-agents:** If a concept involves only 1-2 small files (<5KB each), handle it directly. User-facing questions and checklist presentations always happen in the main context.

---

## Edge Cases

**Scratchpads and working notes.** Discussion files with language like "rough idea", "TBD", "scratchpad", "brainstorm", or "thinking out loud" are not final decisions. Flag these in Phase 1.4 and let the user decide what to integrate.

**Comparison/analysis documents.** Files comparing options (e.g., "Prefect vs Temporal") — integrate the DECISION (which was chosen and why), not the full comparison analysis. Unless the user wants the analysis preserved.

**New core documents.** If discussions introduce a new subsystem with no existing core doc, create one following the conventions of existing core docs (same header structure, same section patterns). Confirm scope with the user.

**Deprecation/removal.** If discussions remove or deprecate something, update all core docs that reference it. Ask the user whether to remove deprecated content entirely or mark it as deprecated.

**Conflicting discussions.** If two discussion files disagree (e.g., one says "use Redis" and another says "use Memcached"), do not pick one. Present both to the user with the relevant quotes and ask them to decide.

**Large integration guides.** Some discussion files may be detailed integration guides with step-by-step implementation instructions. These should be distilled into design specifications (the "what" and "why"), not copied verbatim as implementation steps (the "how"). The core docs are design authority, not implementation guides.
