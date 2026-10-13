-- 004_retention.sql — raw data retention policy
--
-- Architecture: Level 3 Plant Historian (Tools #4046, #5542)
--
-- Raw samples: 90 days retention.
-- Long enough for incident investigation at full resolution and quarterly review;
-- bounded to control storage growth.
--
-- ORDER MATTERS: Apply only after continuous aggregates (002_continuous_aggregates.sql)
-- have been created so downsampled history is preserved.
--
-- Idempotent: safe to re-run.

SELECT add_retention_policy(
    'tag_sample',
    INTERVAL '90 days',
    if_not_exists => TRUE
);

-- ---------------------------------------------------------------------------
-- Rollback / Down note
-- ---------------------------------------------------------------------------
-- To remove the retention policy without data loss:
--   SELECT remove_retention_policy('tag_sample', if_exists => TRUE);
