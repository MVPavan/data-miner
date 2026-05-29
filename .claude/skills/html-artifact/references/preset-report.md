# Preset: report

For plans, status updates, post-mortems, architecture docs, design specs, retrospectives — anywhere a reader expects to skim by section.

## Structure

A report has three layers:

1. **Orientation at the top.** Title, one-line subtitle, and a row of KPI tiles. The reader should answer "what is this and is it current?" in five seconds.
2. **Numbered sections in the middle.** Four to seven sections, each with a clear scannable header. The reader should be able to jump to the section they care about.
3. **Source attribution at the bottom.** A small grey footer listing where the doc came from and when it was generated.

## Section order by report type

The order matters more than people realise — reports of the same type should feel similar. Pick from these defaults and only deviate with reason:

- **Implementation plan**: Milestones · Data flow · Mockups · Key code · Risks · Open questions
- **Status report**: KPIs · Highlights · Shipped · In flight · Blocked/slipped · Sources
- **Post-mortem**: Summary · Timeline · Root cause · Customer impact · Follow-ups · Sources
- **Architecture doc**: Context · System diagram · Components · Data flow · Failure modes · Capacity
- **Design spec**: Problem · Constraints · Proposal · Alternatives · Open questions

## What each section looks like

**Risks / open questions** are tables, not bullet lists. Three columns: the thing, its severity, and the mitigation. Severity uses the severity tag pill. Tables convey shape — "five mediums and one high" reads instantly from a column.

**Key code** is in fenced figures: a muted caption above naming the file path, the code below in monospace, and a small "Copy" button. Avoid pasting in long blocks — pick the 5-20 lines a reader would actually want to copy.

**Mockups** are sketched in HTML using the page's design language — not embedded screenshots. They should feel like part of the doc, not pasted in. Use the surface, border, and muted-text colors for a "sketch" feel.

**Data flow diagrams** are inline SVG. Boxes for stages, arrows between them, dashed lines for asynchronous paths. Use the page's color tokens so the diagram doesn't visually clash.

## Acceptable deviations

- Drop the KPI row if there are genuinely no numeric facts (rare — even "last updated" works).
- Sticky TOC sidebar on viewports >1080px when the doc has 8+ sections.
- Two-column layout for sections where a diagram pairs tightly with its caption.
