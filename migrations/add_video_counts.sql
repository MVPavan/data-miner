-- Migration: Add video count columns to projects table
-- Run this on existing databases before updating the code

ALTER TABLE projects ADD COLUMN IF NOT EXISTS total_videos INTEGER DEFAULT 0;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS videos_pending INTEGER DEFAULT 0;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS videos_downloaded INTEGER DEFAULT 0;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS videos_extracted INTEGER DEFAULT 0;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS videos_failed INTEGER DEFAULT 0;

-- Backfill existing data (optional - monitor will update on next run)
-- Logic: cumulative counts where downloaded includes extracted
UPDATE projects p SET
    total_videos = counts.total,
    videos_pending = counts.pending,
    videos_downloaded = counts.downloaded,
    videos_extracted = counts.extr,
    videos_failed = counts.failed
FROM (
    SELECT 
        pv.project_id,
        COUNT(*)::INTEGER as total,
        COUNT(*) FILTER (WHERE v.status IN ('PENDING', 'DOWNLOADING'))::INTEGER as pending,
        COUNT(*) FILTER (WHERE v.status IN ('DOWNLOADED', 'EXTRACTING', 'EXTRACTED'))::INTEGER as downloaded,
        COUNT(*) FILTER (WHERE v.status = 'EXTRACTED')::INTEGER as extr,
        COUNT(*) FILTER (WHERE v.status = 'FAILED')::INTEGER as failed
    FROM project_videos pv
    JOIN videos v ON pv.video_id = v.video_id
    GROUP BY pv.project_id
) counts
WHERE p.project_id = counts.project_id;
