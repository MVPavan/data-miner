# Why CVAT (and not Label Studio or FiftyOne)

Short version, with sources, so future-you doesn't relitigate.

## vs Label Studio

LS got us off the ground but the failure modes piled up:
- **LSF region-ID quirks** (task 348 broke on colons in `image_id`).
- **Smart-tool dispatcher fragility** (smart_track hijacking smart_visual
  via context-result ordering — see
  [feedback_smart_track_priority](file:///root/.claude/projects/-media-data-2-vlm-code-data-miner/memory/feedback_smart_track_priority.md)).
- **No collaboration model** in Community — assignment is per-task only,
  no project-level views, reviewer UX described as "challenging".
- **Password-on-create silently dropped** by LS Community 1.23 → had to
  call Django ORM.
- **Webhooks don't fire reliably** in Community → forced 5-min cron pull.

CVAT addresses all of the above:
- Native Project → Task → Job model with per-Job assignee.
- No XML labelling-config to maintain.
- API accepts passwords on creation.
- Datumaro/COCO export is JSON, parseable in ~50 lines.
- Reviewer UI is purpose-built keyboard-driven CV bbox tool.

Sources:
- [docs.cvat.ai jobs page](https://docs.cvat.ai/docs/workspace/jobs-page/)
- [CVAT vs LS comparison](https://www.cvat.ai/resources/blog/cvat-or-label-studio-which-one-to-choose)
- Internal: previous LS bugs in commit history of
  [manual_reviewer/](../../manual_reviewer/).

## vs FiftyOne (OSS)

FiftyOne is a curation/QA tool, not an annotation tool. The OSS App is
explicitly single-user (no auth, no concurrent edits). Multi-user
collaboration requires FiftyOne Enterprise, which is sales-quoted and
on-prem-deployable but overkill for 5 named users.

The textbook combo pattern is **CVAT for reviewers + FiftyOne for the
developer's curation dashboard** — they integrate via
`dataset.annotate(backend="cvat")`. We're not blocked on this; bolt
FiftyOne on later if a curation dashboard becomes useful.

Source:
- [docs.voxel51.com Teams overview](https://docs.voxel51.com/teams/overview.html)
- [docs.voxel51.com CVAT integration](https://docs.voxel51.com/integrations/cvat.html)

## What we lose vs LS

- Built-in SAM-based smart_click and our SAM 3.1 tracker. CVAT has its own
  SAM/SAM2 integration via Nuclio, but enabling that brings up another
  service stack we don't need yet. Reviewers said to drop smart features —
  matches their preference.
- Existing 24-class palette XML — replaced with CVAT Project labels (one-time
  port, no ongoing maintenance).
- Familiarity: reviewers will need a 30-min onboarding session.

## What's deferred

- CVAT Nuclio integration (smart tools).
- FiftyOne as a curation/QA dashboard.
- HTTPS termination — currently HTTP-only on `:8081`. Add traefik TLS
  cert config when we expose this beyond the lab network.
