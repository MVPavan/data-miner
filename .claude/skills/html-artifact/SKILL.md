---
name: html-artifact
description: Create polished single-file HTML artifacts as an alternative to markdown for documents meant to be read by humans. Use whenever the user asks for an implementation plan, status report, post-mortem, architecture doc, scaling analysis, capacity dashboard, research explainer, concept walkthrough, PR writeup, or other long-form documents that benefit from richer structure than markdown — KPI tiles, charts, tables, mockups, code blocks with copy buttons, collapsibles, SVG diagrams, or small interactive controls. Also use when the user asks for "an HTML file", "HTML artifact", "HTML doc", "HTML version of this", or "a one-pager". Do NOT use for React components, web apps, marketing pages, or visually expressive design (use frontend-design); for READMEs, docstrings, agent prompts, CLAUDE.md, or anything destined for git/CLI/another agent (use markdown); or for content under ~50 lines where visual structure doesn't earn its weight.
---

# html-artifact

Standalone single-file HTML documents that replace what would otherwise be a long markdown file. Goal: readability and credibility — Stripe docs, not portfolio site.

## When to use

**HTML when all hold:** output is for humans (not another agent or pipeline); visual structure adds information (tables, diagrams, mockups, charts, severity colors, interactive controls); content is ~50+ lines; user wants to share or reference, not hand-edit.

**Markdown when any hold:** README, CLAUDE.md, agent prompt, docstring; lives in git and reviewed in PR diffs; ingested by another LLM, RAG, or eval system; destined for a markdown-native surface (GitHub, Linear, Slack); too short for scaffolding to pay off.

**frontend-design instead when:** the aesthetic goal is "memorable / bold" rather than "readable / credible".

If unsure, ask before producing.

## Standalone-file invariants

Single self-contained `.html` that works double-clicked. Non-negotiable:

- Viewport meta for mobile responsiveness
- Inline CSS, inline JS, no build step
- Dark mode support via `prefers-color-scheme`
- No external trackers, no analytics, no fetches to third-party APIs
- Works on `file://` — no same-origin fetches, no service workers
- System fonts only — no Google Fonts
- Collapses to single column under 720px
- Readable with JS disabled — interactivity is progressive

## Aesthetic direction

The aesthetic is "credible documentation", not "marketing site". Think Stripe docs, Linear changelog, a well-formatted internal wiki page.

**Color:** Neutral page background — off-white in light mode, near-black in dark mode. One accent color used sparingly for emphasis (a single primary KPI, a copy button, the active tab in a tabset). Reserve red/amber/green for severity tags only. Avoid loud color as a background; the page should look like a document, not an ad.

**Typography:** System font stack. Modular type scale — small caption text, body, large section titles, and one or two hero numbers per artifact. Generous line-height in body (1.5–1.7). Tabular numerals for any column of figures.

**Spacing:** Consistent rhythm based on a small set of values (multiples of 4px or 8px). Sections separated by larger gaps than paragraphs; KPI tiles separated by smaller gaps. The same artifact should never mix arbitrary px values — pick a scale and stick to it.

**Layout:** Cap content width for prose readability. Sections are vertically stacked; multi-column only where it earns its weight (KPI rows, chart grids, controls/preview splits). Always single column under 720px.

**Dark mode:** Same layout, inverted palette. The accent and severity colors usually shift to slightly lighter/desaturated variants for readability against dark backgrounds.

## Component vocabulary

Every artifact draws from this small set. Implement each fresh, in keeping with the aesthetic direction above.

**KPI tile row.** 3–5 cards at the top. Each shows a big number, a small muted label below, and optionally a small delta. The fastest way to orient a reader. Use it whenever there are concrete facts the reader needs before reading anything else.

**Numbered section header.** A large grey number ("01", "02") set alongside the section title, with a thin horizontal rule below. Signals "this is scannable, you can jump by section".

**Code block with copy button.** Monospace pre/code with a small "Copy" button in the top-right that writes to the clipboard and briefly flashes "Copied". A muted caption above naming the file path.

**Severity tag.** A small rounded pill in high/med/low severity colors. Tinted background, full color foreground — never raw severity color as a large background. For risks, incidents, statuses.

**Collapsible.** Native `<details>/<summary>` so JS-disabled readers still get the content. For deep-dives, raw data behind charts, anything off the main path.

**Inline SVG diagram.** Real SVG using the page's color tokens — not ASCII, not unicode characters, not embedded screenshots. For flows, architectures, lifecycles. Should feel like part of the doc, not pasted in.

## Presets

Pick exactly one. Read the matching reference before writing.

- **report** — plans, status updates, post-mortems, architecture docs, design specs → `references/preset-report.md`
- **dashboard** — chart-first analyses, capacity, scaling studies, comparisons → `references/preset-dashboard.md`
- **explainer** — "how X works", concept teaching, research summaries → `references/preset-explainer.md`
- **editor** — tuners, config editors, triage boards, anything with "copy back to prompt" → `references/preset-editor.md`

## Anti-patterns

- Gradient hero with centered headline and CTA — the universal "AI website" tell
- Glassmorphism, neon, "shadcn card everywhere" — reports aren't landing pages
- Google Fonts — system fonts are faster and more credible
- Walls of `<p>` with no scaffolding — that's why we picked HTML
- Importing chart libraries for three bars — write inline SVG
- Animations on initial render — doc must be readable instantly
- Buttons that look tappable but do nothing — render as text if non-interactive
- Lorem ipsum, invented metrics, fake names or timestamps
- Mixing arbitrary pixel values — pick a spacing scale and stay on it

## Output protocol

Save to `/mnt/user-data/outputs/` with a descriptive filename. Call `present_files`. Don't paste HTML source into chat — one or two sentences of context is enough.
