# Preset: editor

Interactive throwaway editors: prompt tuners, config editors, triage boards, calibration tools, annotation interfaces. The reader's job is to do something and export the result.

## Hard rule

Every editor ends with at least one export button. Output is paste-ready: JSON, markdown, prompt, YAML — whatever the user feeds back to the agent or commits. Without export, the work disappears.

## When editor vs other presets

Signal phrases: "tune", "tweak", "reorder", "tag", "approve/reject", "calibrate", "drag", "pick a value for". If the user just wants to read, use `report` or `explainer`.

## Structure

Title and a one-line "what this lets you do" subtitle. Optionally a 2-3 line how-to aside.

The main shell is two columns above a 720px viewport, stacked below: controls on the left (inputs, knobs, drag targets), live preview on the right.

At the bottom, an export bar — sticky to the viewport bottom so the button never scrolls out of reach. The export bar has the primary export button (accent color, "Copy as JSON" or whatever), secondary exports next to it, and a Reset button on the side. Always include a small flash confirmation that says "Copied" briefly when the button is clicked.

## State

Plain object in JS, single render function that reads state and writes the DOM, a setState helper that patches state and re-renders. No frameworks. For larger editors (~200+ lines of state logic), split render into renderControls/renderPreview/renderExport so you only call what changed.

## Common patterns

**Live template preview** — editable textarea left, rendered output for several sample inputs right. Highlight variable slots. Pre-fill with a sensible default; never start empty.

**Drag-to-reorder buckets** — columns with draggable cards using the HTML5 drag-and-drop API. Export emits final ordering grouped by bucket.

**Config editor with validation** — fields grouped by area, inline warnings on broken constraints (e.g. flag with prerequisite off). Export a *diff* of changed keys, not full config.

**Calibration / picker** — for values painful to express in text (color, easing curve, cron schedule, threshold values). Sliders + numeric display + live preview. SigLIP2 threshold tuning and hard-negative margin conditions fit here: sliders left, live detection preview right, copy thresholds as YAML at bottom.

## Don'ts

- No "save" button — there's no backend; "save" with no destination is a lie. Use "copy" or "download".
- No localStorage — artifacts are throwaway, persistence confuses.
- No login screen, ever.
- No fetches to external APIs unless explicitly requested.
- No 200KB script for a 10-card triage board.

## Acceptable deviations

- Alpine.js or Vue via CDN for complex editors (mention the cost in the response).
- Keyboard shortcuts in addition to visible buttons. Never keyboard-only.
- "Last export" details block so the user can review what they copied without re-clicking.
