# Long-term vision — permanent multi-team annotation tool

This is the strategic decision record for choosing **vanilla CVAT +
Nuclio** as the foundation for the team's permanent annotation tool,
over alternatives evaluated 2026-05-04. Capture it here so future
contributors don't re-litigate the choice from scratch.

---

## Charter

A self-hosted annotation + review tool that:

- Serves multiple teams (current 5-reviewer Datatang group + others
  to come).
- Provides CVAT-grade UX: Project → Task → Job hierarchy, multi-user
  assignment, three-stage review workflow (annotation → validation →
  acceptance), customizable hotkeys, polished bbox + polygon + mask
  editor.
- Integrates our SAM 3.1 service for live in-editor smart tools:
  `smart_click` (click→mask), `smart_search` (text→detect),
  `smart_visual` (exemplar→similar), `smart_track` (cross-frame video
  propagation).
- Round-trips with `pipeline.db` (seeded predictions in, corrected
  truth out as `Stage.HUMAN_REVIEW`).
- Captures every annotation change for an audit trail.
- Onboards new hires and new teams via public docs, not tribal
  knowledge.
- Is maintainable for 3-5 years with bounded engineering effort.

---

## Options considered

### 1. Modify Label Studio UI (rejected)

Add right-click smart-action menus, custom hotkeys, polish drafts UX
on top of LS Community.

**Why rejected:** the UX gap reviewers feel is structural — LS's flat
single-tier task model and weak multi-user review workflow. Patching
the editor doesn't fix the bones. We'd reinvent CVAT's review hierarchy
on top of LS — large undertaking with worse output.

### 2. Fork CVAT (rejected)

Take cvat-ai/cvat, build smart tools directly into the React frontend,
add a draft annotations tier, replace Nuclio with a direct HTTP backend.

**Why rejected for permanent tool:**

- ~13 person-weeks to 1.0.
- ~1 wk/quarter rebase tax against the **hottest upstream files**
  (`tools-control.tsx`, `controls-side-bar.tsx`, `canvasView.ts`,
  `lambda_manager/views.py`, `engine/models.py` all see frequent
  upstream changes — release every ~10 days, ~17 commits/week).
- Frontend stack is dated: Redux 4.1 + thunk (no RTK Query), Webpack
  (no Vite), React Router 5, antd 5.17, bespoke SVG.js canvas.
- Tribal knowledge: when the maintainer leaves, the next engineer
  inherits a divergent fork, not a public-docs codebase.
- 3-year cost ~31 wks vs ~9 wks for vanilla; 5-year cost ~46 wks vs
  ~11 wks.

A fork can deliver a slightly better UX (right-click smart actions on
existing bboxes, a true draft tier) but the recurring cost is too high
for a multi-team permanent tool.

### 3. Vanilla CVAT + Nuclio + custom functions (chosen)

Stand up upstream CVAT unchanged. Wrap our SAM 3.1 service in
Nuclio functions deployed alongside CVAT. Smart tools land in CVAT's
existing AI Tools panel as new Interactor / Detector / Tracker
selections.

**Why chosen:**

- ~6 person-weeks to 1.0 (3 of those are smart-tool wrappers, 2 are
  pipeline integration, 1 is install/smoke test).
- **Custom code lives in a separate repo** (this one) — zero rebase
  tax against CVAT.
- CVAT upstream improvements flow in via `docker pull` — SAM4, new
  trackers, UI improvements all ride free.
- Three of our four smart tools are already half-built upstream:
  - `smart_click` → CVAT ships SAM/SAM2/SAM3 click interactors. We
    swap the bundled model for our fine-tuned SAM 3.1.
  - `smart_search` → CVAT's SAM3 supports "label-as-text-prompt"
    mode. We may use it as-is or wrap our text_detect.
  - `smart_track` → CVAT ships a SAM2 Tracker. We can use it as-is
    or swap for our SAM 3.1 video mode if needed.
  - `smart_visual` → only net-new function we must write.
- Public docs onboard new hires (`docs.cvat.ai`).
- License: CVAT MIT. Nuclio Apache-2.0. Both fine for commercial use.

**Trade-offs accepted:**

- No first-class "draft tier" (yellow predictions). Mitigation: use
  CVAT's three-stage workflow — seeded predictions land in
  `annotation` stage, reviewer edits and advances to `validation`.
  Arguably more rigorous than yellow drafts.
- No annotation-level webhooks. Mitigation: cron-poll EventsAPI
  CSV export, same pattern as our LS [sync_ls_to_disk.py](../../manual_reviewer/scripts/sync_ls_to_disk.py)
  cron today.
- No right-click smart-action on existing bboxes. Mitigation:
  smart_visual on a redrawn exemplar serves the same flow. Re-evaluate
  if reviewers complain after 2 weeks of real use.
- Nuclio deployment has known papercuts (`nuctl` version pinning,
  separate compose file). Bounded — package once, deploy many times.

---

## Recurring-cost comparison

| | LS UI mods | CVAT fork | **Vanilla CVAT + Nuclio** |
|---|---|---|---|
| Upfront effort | 4-6 wks | 13 wks | **~6 wks** |
| Annual maintenance | ~2 wks | 4-6 wks rebase | **~1 wk** |
| 3-year cost | ~22 wks | ~31 wks | **~9 wks** |
| 5-year cost | ~30 wks | ~46 wks | **~11 wks** |
| Skill required | React+MobX + LS internals | React+Redux+Django+SVG.js | Python (Nuclio fns) |
| New-hire ramp | Tribal | Tribal | **Public docs** |
| Resilience to maintainer churn | Low | Lowest | **Highest** |

---

## Re-evaluate the choice if

- CVAT upstream becomes unmaintained, gets sold to a hostile vendor,
  or relicenses away from MIT.
- Reviewers tell us after 6+ months of real use that the UX gaps
  (no right-click smart action, no draft tier) are daily friction
  points — at which point a small targeted fork from a position of
  strength becomes reasonable.
- A new tool emerges that combines CVAT's UX with a simple HTTP
  ML-backend protocol (no Nuclio packaging). Unlikely but worth
  watching.

Until one of those happens, do not re-open this decision.
