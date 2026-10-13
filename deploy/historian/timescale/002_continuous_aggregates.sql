-- 002_continuous_aggregates.sql — downsampled rollups for plant history
--
-- Architecture: Level 3 Plant Historian (Tools #4046, #5542)
--
-- Continuous aggregates replace destructive byte-cap deletion with downsampled persistence:
-- raw data ages out after 90 days, while 1-minute rollups survive for 2 years.
--
-- Carries min, max, avg, sum, and count:
-- - min and max preserve excursions (an averaged peak is invisible in safety reviews).
-- - sum and count enable accurate rollups across varying sample counts.
--
-- Idempotent: safe to re-run.

-- --------------------------------------------------------------- 1 minute ---

CREATE MATERIALIZED VIEW IF NOT EXISTS tag_sample_1m
WITH (timescaledb.continuous) AS
SELECT
    time_bucket(INTERVAL '1 minute', ts) AS bucket,
    tag_id,
    avg(value)   AS avg_value,
    min(value)   AS min_value,
    max(value)   AS max_value,
    sum(value)   AS sum_value,
    count(*)     AS sample_count
FROM tag_sample
GROUP BY bucket, tag_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'tag_sample_1m',
    start_offset      => INTERVAL '1 hour',
    end_offset        => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists     => TRUE
);

CREATE INDEX IF NOT EXISTS ix_tag_sample_1m_tag_bucket
    ON tag_sample_1m (tag_id, bucket DESC);

COMMENT ON MATERIALIZED VIEW tag_sample_1m IS
    'One-minute rollup. Retained 2 years. Query this view rather than tag_sample '
    'for ranges beyond the 90-day raw retention window.';

-- ---------------------------------------------------------------------------
-- Rollback / Down note
-- ---------------------------------------------------------------------------
-- To remove the continuous aggregate policy and materialized view:
--   SELECT remove_continuous_aggregate_policy('tag_sample_1m', if_exists => TRUE);
--   DROP MATERIALIZED VIEW IF EXISTS tag_sample_1m;
