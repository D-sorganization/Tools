-- 003_compression.sql — columnar compression on aged chunks
--
-- Architecture: Level 3 Plant Historian (Tools #4046, #5542)
--
-- Segmenting by tag_id groups each tag's samples into a contiguous compressed
-- array, enabling 10-20x compression on slowly-varying float process values.
--
-- Ordering by ts DESC matches dominant query patterns.
-- The 7-day uncompressed window ensures recent data is queried with zero decompression overhead.
--
-- Idempotent: safe to re-run.

ALTER TABLE tag_sample SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'tag_id',
    timescaledb.compress_orderby   = 'ts DESC'
);

SELECT add_compression_policy(
    'tag_sample',
    INTERVAL '7 days',
    if_not_exists => TRUE
);

-- ---------------------------------------------------------------------------
-- Rollback / Down note
-- ---------------------------------------------------------------------------
-- To disable compression and remove the policy without data loss:
--   SELECT remove_compression_policy('tag_sample', if_exists => TRUE);
--   ALTER TABLE tag_sample SET (timescaledb.compress = false);
