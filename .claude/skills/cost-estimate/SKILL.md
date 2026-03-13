---
name: cost-estimate
description: >
  Estimate the development cost of a codebase based on lines of code, complexity, market rates,
  and team composition. Use this skill whenever the user asks about how much a codebase would
  cost to build, what a project is worth, development time estimates, engineering cost analysis,
  freelancer/contractor pricing for a project, or ROI of AI-assisted development. Also trigger
  when the user says things like "how much would this cost to build from scratch", "estimate dev
  hours", "what's this codebase worth", "calculate development cost", "how long would this take
  a team to build", or "what would I pay a developer for this". Even if the user just asks
  "how much is this worth" in the context of a codebase, use this skill.
---

# Cost Estimate Skill

Estimate the development cost of a codebase by analyzing its size, complexity, and specialized
technology requirements, then mapping those to market rates and realistic team structures.

## Overview

This skill walks through 7 steps:

1. **Analyze the Codebase** — count lines, identify complexity factors
2. **Calculate Development Hours** — convert LOC to hours using productivity benchmarks
3. **Research Market Rates** — look up current hourly rates via web search
4. **Calculate Organizational Overhead** — convert raw hours to realistic calendar time
5. **Calculate Full Team Cost** — add supporting roles (PM, design, QA, etc.)
6. **Generate the Cost Estimate Report** — produce a polished, shareable document
7. **Calculate Claude ROI** (optional) — if the project was built with AI assistance, measure value per Claude hour

Read `references/rates-and-multipliers.md` for all the detailed lookup tables (productivity
rates, overhead factors, team multipliers, role ratios). Read `references/output-template.md`
for the full report template to fill in.

---

## Step 1: Analyze the Codebase

Systematically scan the entire codebase to build a complete picture. Use `find`, `wc -l`,
`cloc` (if available), or manual counting to gather:

- **Total lines of code** broken down by language (Swift, Python, TypeScript, C++, etc.)
- **Test lines** (files in `test/`, `tests/`, `__tests__/`, `spec/`)
- **Documentation lines** (markdown, comments, READMEs)
- **Config & build lines** (CI/CD, Dockerfiles, build scripts)

Identify complexity factors:

- System-level programming (kernel extensions, camera extensions, DAL plugins, drivers)
- GPU/shader programming (Metal, CUDA, OpenGL, Vulkan)
- Native interop (C++ bridging, FFI, JNI)
- Audio/video processing (AVFoundation, CoreMedia, FFmpeg, WebRTC)
- Third-party AI/ML integrations (OpenAI, CoreML, TensorFlow)
- Cryptography or security-sensitive code
- Complex networking (WebSocket, gRPC, custom protocols)

These factors directly influence the productivity rates used in Step 2.

## Step 2: Calculate Development Hours

Consult `references/rates-and-multipliers.md` § "Hourly Productivity Estimates" for the
per-category line rates.

**Process:**

1. Classify each source file into a complexity category (simple UI, business logic, GPU, etc.)
2. Divide lines by the appropriate productivity rate to get **base coding hours**
3. Apply overhead multipliers from § "Additional Time Factors":
   - Architecture & design: +15–20%
   - Debugging & troubleshooting: +25–30%
   - Code review & refactoring: +10–15%
   - Documentation: +10–15%
   - Integration & testing: +20–25%
   - Learning curve (specialized tech): +10–20%
4. Sum to get **Total Estimated Development Hours**

Be transparent about which multipliers you chose and why. If the codebase is heavily
specialized (e.g., Metal shaders + CoreMediaIO), use the higher end of ranges.

## Step 3: Research Market Rates

Use web search to find current-year hourly rates. Search for:

- `"senior full stack developer hourly rate [current year]"`
- `"[primary language] developer contractor rate [current year]"`
- `"senior software engineer hourly rate United States [current year]"`
- `"[specialty] developer freelance rate [current year]"`

Produce three tiers: **Low**, **Average**, **High**. Note that specialized skills
(macOS system extensions, GPU programming, video pipelines) command premium rates —
typically 20–40% above general full-stack rates.

## Step 4: Calculate Organizational Overhead

Real developers don't code 40 hours per week. Consult
`references/rates-and-multipliers.md` § "Organizational Overhead" for the weekly time
breakdown and efficiency factors by company type.

**Formula:**
```
Calendar Weeks = Total Dev Hours ÷ (40 × Efficiency Factor)
```

Present a table showing calendar time across company types:
Solo/Startup (65%), Growth (55%), Enterprise (45%), Large Bureaucracy (35%).

## Step 5: Calculate Full Team Cost

Engineering doesn't ship alone. Consult `references/rates-and-multipliers.md`
§ "Full Team Cost" for role ratios and multipliers.

**Formula:**
```
Full Team Cost = Engineering Cost × Team Multiplier
```

Team multipliers: Solo (1.0×), Lean Startup (1.45×), Growth (2.2×), Enterprise (2.65×).

Provide a role-by-role breakdown for at least the "Growth Company" scenario (PM, Design,
EM, QA, PgM, Docs, DevOps) with hours and rates.

## Step 6: Generate the Cost Estimate Report

Read `references/output-template.md` and fill in every section. The report should be
professional, shareable with stakeholders, and include:

- Codebase metrics (LOC by language, complexity factors)
- Development time estimates with overhead breakdown
- Calendar time table across company types
- Market rate research with sources
- Engineering-only cost table (low/avg/high)
- Full team cost table across company stages
- Role breakdown for Growth Company
- Grand total summary
- Assumptions and exclusions

Present the final report as a well-formatted markdown document. If the user wants a
Word doc or PDF, use the appropriate skill to produce it.

## Step 7: Claude ROI Analysis (Optional)

If the project was built with AI assistance (Claude, Copilot, etc.), calculate the
return on investment. This is especially relevant when there's git history available.

### Determine Claude Active Hours

**Method 1 — Git History (preferred):**
```bash
git log --format="%ai" | sort
```
Cluster commits into sessions using 4-hour windows:
- 1–2 commits in a window → ~1 hour
- 3–5 commits → ~2 hours
- 6–10 commits → ~3 hours
- 10+ commits → ~4 hours

**Method 2 — File Timestamps:**
```bash
find . -name "*.swift" -o -name "*.py" -o -name "*.ts" | xargs stat --format="%Y" | sort
```
Apply the same session clustering.

**Method 3 — Fallback (LOC-based):**
```
Claude Active Hours ≈ Total LOC ÷ 350
```

### Calculate ROI Metrics

```
Value per Claude Hour = Total Code Value ÷ Claude Active Hours
Speed Multiplier = Human Dev Hours ÷ Claude Active Hours
Savings = (Human Hours × Avg Rate) - Claude Subscription Cost
ROI = Savings ÷ Claude Cost
```

Include a headline summary: *"Claude worked for approximately X hours and produced
the equivalent of $Y in professional development value — roughly $Z per Claude hour."*

---

## Key Principles

- **Be conservative** — it's better to slightly overestimate than underestimate cost.
  Stakeholders prefer "under budget" to "over budget."
- **Show your work** — every number should trace back to a productivity rate, market
  data point, or stated assumption.
- **Acknowledge uncertainty** — use ranges rather than single-point estimates where possible.
- **Tailor to context** — if the user mentions their company stage, emphasize that
  scenario. If they're a solo founder, lead with the lean numbers.
- **Include caveats** — the estimate doesn't cover marketing, legal, hosting,
  infrastructure, or ongoing maintenance unless explicitly requested.
