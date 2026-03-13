# Rates and Multipliers Reference

All lookup tables for the cost-estimate skill. Organized by section for quick reference.

## Table of Contents

1. [Hourly Productivity Estimates](#hourly-productivity-estimates)
2. [Additional Time Factors](#additional-time-factors)
3. [Organizational Overhead](#organizational-overhead)
4. [Full Team Cost](#full-team-cost)

---

## Hourly Productivity Estimates

Based on industry standards for a **senior full-stack developer** (5+ years experience).
These are lines of code per hour of focused coding time.

| Code Category | Lines/Hour | When to Apply |
|---|---|---|
| Simple CRUD / UI code | 30–50 | Standard forms, layouts, basic API calls |
| Complex business logic | 20–30 | State machines, validation, algorithms |
| GPU / Metal / shader programming | 10–20 | Metal shaders, CUDA kernels, compute pipelines |
| Native C++ interop / FFI | 10–20 | Bridging headers, JNI, unsafe Rust FFI |
| Video / audio processing | 10–15 | AVFoundation, CoreMedia, FFmpeg, WebRTC |
| System extensions / plugins | 8–12 | Camera extensions, DAL plugins, kernel extensions |
| Comprehensive tests | 25–40 | Unit tests, integration tests, snapshot tests |
| Infrastructure / CI / config | 30–50 | Dockerfiles, GitHub Actions, Makefiles |
| Documentation (prose) | 40–60 | READMEs, API docs, architecture docs |

**How to apply:** Classify each source file (or group of files) into a category, then
divide its line count by the corresponding rate to get base coding hours for that file.

---

## Additional Time Factors

These are overhead multipliers applied **on top of** base coding hours. They account
for the non-coding work that's essential to producing production-quality software.

| Factor | Multiplier Range | Rationale |
|---|---|---|
| Architecture & design | +15–20% | System design, API design, data modeling before coding |
| Debugging & troubleshooting | +25–30% | Finding and fixing bugs, investigating issues |
| Code review & refactoring | +10–15% | Giving and receiving reviews, cleanup passes |
| Documentation | +10–15% | Inline comments, READMEs, architecture decision records |
| Integration & testing | +20–25% | Wiring components together, CI setup, end-to-end testing |
| Learning curve (new frameworks) | +10–20% | Ramp-up time for specialized tech (Metal, CoreMediaIO, etc.) |

**How to apply:** Sum the percentages you select (typically 100–135% total overhead)
and multiply by base coding hours.

**Example:** 1,000 base hours × (1 + 0.17 + 0.27 + 0.12 + 0.12 + 0.22 + 0.15) = 1,000 × 2.05 = 2,050 total hours

**Guidance on choosing within ranges:**
- Use the **low end** for well-understood domains with mature tooling
- Use the **mid range** for typical professional projects
- Use the **high end** for novel technology, poor documentation, or first-time frameworks

---

## Organizational Overhead

Real companies have meetings, code reviews, communication overhead, and context switching.
This section converts raw development hours to realistic calendar time.

### Weekly Time Allocation (Typical Full-Time Developer)

| Activity | Hours/Week | Notes |
|---|---|---|
| Pure coding time | 20–25 | Actual focused development |
| Daily standups | 1.25 | 15 min × 5 days |
| Weekly team sync | 1–2 | All-hands, team meetings |
| 1:1s with manager | 0.5–1 | Weekly or biweekly |
| Sprint planning / retro | 1–2 | Per week average |
| Code reviews (giving) | 2–3 | Reviewing teammates' PRs |
| Slack / email / async comms | 3–5 | Communication overhead |
| Context switching | 2–4 | Interruptions, task switching |
| Ad-hoc meetings | 1–2 | Unplanned discussions |
| Admin / HR / tooling | 1–2 | Timesheets, access requests, tool setup |

### Coding Efficiency by Company Type

| Company Type | Efficiency Factor | Coding Hrs/Week | Description |
|---|---|---|---|
| Solo / Startup (lean) | 65% | ~26 hrs | Minimal process, maximum autonomy |
| Growth Company | 55% | ~22 hrs | Some process, growing team coordination |
| Enterprise | 45% | ~18 hrs | Significant process, compliance, approvals |
| Large Bureaucracy | 35% | ~14 hrs | Heavy process, many stakeholders, slow decisions |

### Calendar Time Formula

```
Calendar Weeks = Total Dev Hours ÷ (40 × Efficiency Factor)
Calendar Months = Calendar Weeks ÷ 4.33
Calendar Years = Calendar Months ÷ 12
```

**Example:** 3,288 total dev hours at 55% (Growth Company):
- Calendar weeks = 3,288 ÷ 22 = 149.5 weeks
- Calendar months = 149.5 ÷ 4.33 ≈ 34.5 months
- Calendar years ≈ 2.9 years

---

## Full Team Cost

Engineering doesn't ship products alone. These tables help estimate the cost of the
full team needed to support a software product.

### Supporting Role Ratios

Each role's hours expressed as a fraction of engineering hours.

| Role | Ratio to Eng Hours | Typical Hourly Rate | What They Do |
|---|---|---|---|
| Product Management | 0.25–0.40× | $125–200/hr | PRDs, roadmap, stakeholder management |
| UX/UI Design | 0.20–0.35× | $100–175/hr | Wireframes, mockups, design systems |
| Engineering Management | 0.12–0.20× | $150–225/hr | 1:1s, hiring, performance reviews, strategy |
| QA / Testing | 0.15–0.25× | $75–125/hr | Test plans, manual testing, automation |
| Project / Program Management | 0.08–0.15× | $100–150/hr | Schedules, dependencies, status reporting |
| Technical Writing | 0.05–0.10× | $75–125/hr | User docs, API docs, internal docs |
| DevOps / Platform | 0.10–0.20× | $125–200/hr | CI/CD, infrastructure, deployments |

### Team Composition by Company Stage

Percentage of engineering hours allocated to each supporting role.

| Stage | PM | Design | EM | QA | PgM | Docs | DevOps |
|---|---|---|---|---|---|---|---|
| Solo / Founder | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| Lean Startup | 15% | 15% | 5% | 5% | 0% | 0% | 5% |
| Growth Company | 30% | 25% | 15% | 20% | 10% | 5% | 15% |
| Enterprise | 40% | 35% | 20% | 25% | 15% | 10% | 20% |

### Full Team Multipliers

Pre-calculated multipliers to quickly estimate total team cost from engineering cost.

| Company Stage | Team Multiplier | Derivation |
|---|---|---|
| Solo / Founder | 1.0× | Just engineering |
| Lean Startup | ~1.45× | Engineering + lean support |
| Growth Company | ~2.2× | Engineering + full cross-functional team |
| Enterprise | ~2.65× | Engineering + heavy support structure |

### Formula

```
Full Team Cost = Engineering Cost × Team Multiplier
```

**Example:** $500K engineering cost at Growth Company stage = $500K × 2.2 = **$1.1M total team cost**

### Detailed Calculation (Growth Company)

For a more granular breakdown, calculate each role individually:

```
PM Hours      = Engineering Hours × 0.30
Design Hours  = Engineering Hours × 0.25
EM Hours      = Engineering Hours × 0.15
QA Hours      = Engineering Hours × 0.20
PgM Hours     = Engineering Hours × 0.10
Docs Hours    = Engineering Hours × 0.05
DevOps Hours  = Engineering Hours × 0.15

Each Role Cost = Role Hours × Role Hourly Rate
Full Team Cost = Sum of all role costs + Engineering Cost
```
