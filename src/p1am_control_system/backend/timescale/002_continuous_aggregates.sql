-- 002_continuous_aggregates.sql — downsampled rollups
--
-- This is the migration that changes the retention story. Today the SQLite
-- historian enforces a byte cap by DELETING the oldest samples, so a long
-- enough horizon simply loses its history. Here, raw data ages out but
-- aggregates survive: 1-minute resolution for two years, 1-hour indefinitely.
--
-- min and max are carried, not just avg. An averaged excursion is an invisible
-- excursion, and for process safety review the peak is the number that matters.
--
-- sum and count are carried so the hourly rollup can compute a correctly
-- weighted mean. avg(avg) is only right when every bucket has the same sample
-- count, which is exactly what a lossy shipper cannot guarantee.

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

-- ------------------------------------------------------------------ 1 hour ---
-- Hierarchical rollup: built from the 1-minute aggregate rather than from raw
-- samples, so the hourly refresh never rescans the raw hypertable.
-- Requires TimescaleDB 2.9+.

CREATE MATERIALIZED VIEW IF NOT EXISTS tag_sample_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket(INTERVAL '1 hour', bucket) AS bucket,
    tag_id,
    sum(sum_value) / NULLIF(sum(sample_count), 0) AS avg_value,
    min(min_value)   AS min_value,
    max(max_value)   AS max_value,
    sum(sum_value)   AS sum_value,
    sum(sample_count) AS sample_count
FROM tag_sample_1m
GROUP BY 1, 2
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'tag_sample_1h',
    start_offset      => INTERVAL '6 hours',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists     => TRUE
);

CREATE INDEX IF NOT EXISTS ix_tag_sample_1h_tag_bucket
    ON tag_sample_1h (tag_id, bucket DESC);

COMMENT ON MATERIALIZED VIEW tag_sample_1m IS
    'One-minute rollup. Retained 2 years. Query this, not tag_sample, for any '
    'range beyond the raw retention window.';
COMMENT ON MATERIALIZED VIEW tag_sample_1h IS
    'One-hour rollup, built hierarchically from tag_sample_1m. Retained '
    'indefinitely — this is the permanent plant record.';
