-- 003_compression.sql — columnar compression on aged chunks
--
-- segmentby = tag_id is the setting that matters. It groups each tag's samples
-- into one compressed row-array, so a slowly-varying process value compresses
-- against itself rather than against an interleaved neighbour. This is what
-- delivers the 10-20x on float series; getting it wrong (or omitting it) gives
-- closer to 2-3x.
--
-- orderby = ts DESC keeps the newest sample first inside a compressed batch,
-- which is the direction trend queries scan.
--
-- 7 days matches the window in which data is still queried at raw resolution
-- often enough that decompression overhead would be felt.

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

-- The 1-minute aggregate is itself large enough to be worth compressing at
-- longer horizons. The hourly rollup is small and is left uncompressed so the
-- permanent record stays cheap to query.
ALTER MATERIALIZED VIEW tag_sample_1m SET (
    timescaledb.compress = TRUE
);

SELECT add_compression_policy(
    'tag_sample_1m',
    INTERVAL '90 days',
    if_not_exists => TRUE
);
