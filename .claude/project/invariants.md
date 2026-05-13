# Mechanically Checkable Invariants

Status: adopted for v2.7 design docs on 2026-04-07

These are current mechanically checkable project facts for the repo as it exists today.
Promote implementation invariants only after code, manifests, and test config exist.

## [INV-01] Common Design Is The Top Authority

- Statement: `docs/design/v_2_7/core/bodha-design-v2_7.md` is present and marked authoritative.
- Check: `rg -n "^\\*\\*Status:\\*\\* Authoritative$" docs/design/v_2_7/core/bodha-design-v2_7.md`
- Must return: exactly one line
- Why it matters: this is the source of truth when component docs or historical setup files disagree.

## [INV-02] Temporal Replaced The Older Scheduler Model

- Statement: the v2.7 design says Temporal is the sole orchestrator and replaces `async_core.Scheduler`.
- Check: `rg -n "Temporal as sole orchestrator|replaces async_core\\.Scheduler" docs/design/v_2_7/core/bodha-design-v2_7.md docs/design/v_2_7/core/bodha-infrastructure-v2_7.md`
- Must return: one or more matching lines
- Why it matters: the old Bodha setup still assumes the pre-Temporal scheduler model.

## [INV-03] pydantic-settings Replaced OmegaConf

- Statement: the v2.7 design adopts `pydantic-settings` instead of OmegaConf.
- Check: `rg -n "pydantic-settings replaces OmegaConf" docs/design/v_2_7/core/bodha-design-v2_7.md docs/design/v_2_7/core/bodha-infrastructure-v2_7.md`
- Must return: one or more matching lines
- Why it matters: historical Python/config rules that still hard-code OmegaConf are stale.

## [INV-04] PostgreSQL Is The Authoritative Write Boundary

- Statement: Bodha treats PostgreSQL as the authoritative write boundary and source of record.
- Check: `rg -n "Postgres.*(sole authority|source of record|Authoritative write boundary)" docs/design/v_2_7/core/bodha-design-v2_7.md docs/design/v_2_7/core/bodha-chitta-v2_7.md`
- Must return: one or more matching lines
- Why it matters: write-path safety, replayability, and projection discipline all depend on this boundary.

## [INV-05] Tools And Skills Are Registry Infrastructure

- Statement: tools and skills are registry infrastructure, not memory-ledger entries.
- Check: `rg -n "Tools and skills are \\*\\*identity infrastructure\\*\\*" docs/design/v_2_7/core/bodha-tools-skills-v2_7.md`
- Must return: exactly one line
- Why it matters: this keeps capability identity separate from memory storage and Dhṛti write-gate logic.

## [INV-06] Raw Document Chunks Never Enter Citta

- Statement: raw document chunks and verbatim passages do not enter Citta.
- Check: `rg -n "Raw chunks and verbatim passages never enter Citta" docs/design/v_2_7/core/bodha-rag-v2_7.md`
- Must return: exactly one line
- Why it matters: the bookshelf model depends on keeping raw document content external while storing only internalized knowledge and provenance.

## [INV-07] Source Quote Is Mandatory In v2.7 Extraction

- Statement: every ExtractionCandidate must include a `source_quote` field in v2.7.
- Check: `rg -n "Every ExtractionCandidate must include a .source_quote. field|source_quote mandatory" docs/design/v_2_7/core/bodha-design-v2_7.md docs/design/v_2_7/core/bodha-dhriti-v2_7.md`
- Must return: one or more matching lines
- Why it matters: v2.7 extraction grounding depends on verbatim evidence instead of trusting model confidence alone.

## [INV-08] Phase 1.5 Mechanical Verification Is Non-LLM

- Statement: Phase 1.5 grounding verification uses mechanical checks, not LLM calls.
- Check: `rg -n "Phase 1\\.5.*non-LLM|No LLM calls|mechanical verification is non-LLM" docs/design/v_2_7/core/bodha-design-v2_7.md docs/design/v_2_7/core/bodha-dhriti-v2_7.md docs/design/v_2_7/core/bodha-job-plan-v2_7.md`
- Must return: one or more matching lines
- Why it matters: the v2.7 hardening story depends on adding verifiable guardrails rather than another opaque reasoning step.

## [INV-09] Fabricated Candidates Are Rejected Before Phase 2

- Statement: if grounding fails badly enough, the candidate is rejected before escalation or Phase 2.
- Check: `rg -n "Fabricated candidates are rejected before Phase 2|grounding_score < 0\\.5.*REJECT|automatic \\*\\*REJECT\\*\\*" docs/design/v_2_7/core/bodha-design-v2_7.md docs/design/v_2_7/core/bodha-dhriti-v2_7.md`
- Must return: one or more matching lines
- Why it matters: this is the strongest new v2.7 safeguard against fabricated memories entering the write path.
