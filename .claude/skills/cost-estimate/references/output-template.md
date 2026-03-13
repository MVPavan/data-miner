# Output Template

Use this template to structure the final cost estimate report. Replace all `[bracketed]`
placeholders with calculated values. Remove any sections that don't apply (e.g., skip
the Claude ROI section if the project wasn't AI-assisted).

---

## [Project Name] — Development Cost Estimate

**Analysis Date**: [Current Date]
**Codebase Version**: [Version or commit hash if available]

---

### Codebase Metrics

**Total Lines of Code**: [number]

| Language / Category | Lines |
|---|---|
| [Primary language] | [number] |
| [Secondary language] | [number] |
| [Shaders / GPU code] | [number] |
| Tests | [number] |
| Documentation | [number] |
| Config / Build | [number] |

**Complexity Factors**:
- [List each advanced framework, system-level integration, GPU programming requirement, etc.]
- [Note any third-party integrations that add complexity]

---

### Development Time Estimate

**Base Development Hours**: [number] hours

| Code Category | Lines | Rate (lines/hr) | Hours |
|---|---|---|---|
| [Category 1] | [lines] | [rate] | [hours] |
| [Category 2] | [lines] | [rate] | [hours] |
| [Category 3] | [lines] | [rate] | [hours] |
| **Subtotal** | **[total lines]** | | **[total hours]** |

**Overhead Multipliers**:

| Factor | Percentage | Hours Added |
|---|---|---|
| Architecture & Design | +[X]% | [hours] |
| Debugging & Troubleshooting | +[X]% | [hours] |
| Code Review & Refactoring | +[X]% | [hours] |
| Documentation | +[X]% | [hours] |
| Integration & Testing | +[X]% | [hours] |
| Learning Curve | +[X]% | [hours] |
| **Total Overhead** | **+[X]%** | **[hours]** |

**Total Estimated Development Hours**: [number] hours

---

### Realistic Calendar Time

| Company Type | Efficiency | Coding Hrs/Week | Calendar Weeks | Calendar Time |
|---|---|---|---|---|
| Solo / Startup (lean) | 65% | 26 hrs | [X] weeks | ~[X] months |
| Growth Company | 55% | 22 hrs | [X] weeks | ~[X] months |
| Enterprise | 45% | 18 hrs | [X] weeks | ~[X] years |
| Large Bureaucracy | 35% | 14 hrs | [X] weeks | ~[X] years |

---

### Market Rate Research

**Senior Developer Rates ([current year])**:

| Tier | Hourly Rate | Context |
|---|---|---|
| Low | $[X]/hr | Remote, mid-tier market |
| Average | $[X]/hr | Standard US market |
| High | $[X]/hr | SF/NYC, specialized skills |

**Recommended Rate for This Project**: $[X]/hr

*Rationale*: [Explain why — e.g., "This project requires specialized macOS development
skills (CoreMediaIO, Metal, system extensions) which command premium rates."]

---

### Engineering-Only Cost

| Scenario | Hourly Rate | Total Hours | Total Cost |
|---|---|---|---|
| Low-end | $[X] | [hours] | **$[amount]** |
| Average | $[X] | [hours] | **$[amount]** |
| High-end | $[X] | [hours] | **$[amount]** |

**Recommended Range (Engineering Only)**: **$[low] – $[high]**

---

### Full Team Cost

| Company Stage | Team Multiplier | Engineering Cost | Full Team Cost |
|---|---|---|---|
| Solo / Founder | 1.0× | $[X] | **$[X]** |
| Lean Startup | 1.45× | $[X] | **$[X]** |
| Growth Company | 2.2× | $[X] | **$[X]** |
| Enterprise | 2.65× | $[X] | **$[X]** |

**Role Breakdown (Growth Company, Average Rate)**:

| Role | Hours | Rate | Cost |
|---|---|---|---|
| Engineering | [X] hrs | $[X]/hr | $[X] |
| Product Management | [X] hrs | $[X]/hr | $[X] |
| UX/UI Design | [X] hrs | $[X]/hr | $[X] |
| Engineering Management | [X] hrs | $[X]/hr | $[X] |
| QA / Testing | [X] hrs | $[X]/hr | $[X] |
| Project Management | [X] hrs | $[X]/hr | $[X] |
| Technical Writing | [X] hrs | $[X]/hr | $[X] |
| DevOps / Platform | [X] hrs | $[X]/hr | $[X] |
| **TOTAL** | **[X] hrs** | | **$[X]** |

---

### Grand Total Summary

| Metric | Solo | Lean Startup | Growth Co | Enterprise |
|---|---|---|---|---|
| Calendar Time | [X] | [X] | [X] | [X] |
| Total Human Hours | [X] | [X] | [X] | [X] |
| **Total Cost** | **$[X]** | **$[X]** | **$[X]** | **$[X]** |

---

### Claude ROI Analysis

*Include this section only if the project was built with AI assistance.*

**Project Timeline**:
- First commit / project start: [date]
- Latest commit: [date]
- Total calendar time: [X] days ([X] weeks)

**Claude Active Hours Estimate**:
- Total sessions identified: [X]
- Estimated active hours: [X] hours
- Method used: [git clustering / file timestamps / LOC estimate]

**Value per Claude Hour**:

| Value Basis | Total Value | Claude Hours | Value per Claude Hour |
|---|---|---|---|
| Engineering only (avg rate) | $[X] | [X] hrs | **$[X]/hr** |
| Full team — Growth Co | $[X] | [X] hrs | **$[X]/hr** |

**Speed vs. Human Developer**:
- Estimated human hours for same work: [X] hours
- Claude active hours: [X] hours
- **Speed multiplier: [X]×**

**Cost Comparison**:
- Human developer cost: $[X]
- Estimated Claude cost: $[X] (subscription + API)
- **Net savings: $[X]**
- **ROI: [X]×**

*Headline: Claude worked for approximately [X] hours and produced the equivalent of
$[X] in professional development value — roughly **$[X] per Claude hour**.*

---

### Assumptions & Exclusions

**Assumptions**:
1. Rates based on US market averages ([current year])
2. Full-time equivalent allocation for all roles
3. Includes complete implementation of all analyzed features
4. Senior developer (5+ years experience) as baseline

**Not Included**:
- Marketing & sales
- Legal & compliance
- Office space / equipment
- Cloud hosting / infrastructure costs
- Ongoing maintenance post-launch
- Recruitment / hiring costs

---

### Confidence & Caveats

**Confidence Level**: [High / Medium / Low]

**Key Uncertainties**:
- [List any factors that could significantly shift the estimate]
- [Note areas where complexity is hard to assess from code alone]

**Recommendation**: [Any advice — e.g., "For budgeting purposes, use the Growth Company
scenario at average rates as your baseline, and add 15–20% contingency."]
