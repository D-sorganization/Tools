-- 004_retention.sql — age raw data out, keep aggregates
--
-- ORDER MATTERS: these policies drop data. Do not apply this file until
-- 002_continuous_aggregates.sql has been applied AND has actually materialised
-- (check with the verification query at the bottom of README.md). Dropping raw
-- chunks before the aggregates have been built loses that history permanently.
--
-- Contrast with the current SQLite behaviour, where P1AM_HISTORIAN_MAX_BYTES
-- purges oldest rows outright: here the raw resolution ages out but the record
-- survives at reduced resolution, forever.

-- Raw samples: 90 days. Long enough for incident investigation at full
-- resolution and for a quarterly review; short enough to stay affordable.
SELECT add_retention_policy(
    'tag_sample',
    INTERVAL '90 days',
    if_not_exists => TRUE
);

-- 1-minute rollup: 2 years. Covers year-over-year comparison and campaign
-- history at a resolution that still resolves process dynamics.
SELECT add_retention_policy(
    'tag_sample_1m',
    INTERVAL '2 years',
    if_not_exists => TRUE
);

-- 1-hour rollup: NO retention policy, deliberately. This is the permanent
-- plant record. At 10k tags an hourly rollup is ~88M rows/year, which is small.
-- If this ever needs to be bounded, that is a conscious decision to destroy
-- plant history and should be made explicitly, not inherited from a default.
