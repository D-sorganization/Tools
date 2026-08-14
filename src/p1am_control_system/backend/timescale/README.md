# TimescaleDB plant historian schema

Versioned SQL for the Level 3 plant historian. See the epic (#4046) for how this
fits the overall architecture.

**These migrations are not applied automatically.** The application never runs
DDL against the historian at startup. Schema changes on a plant historian are an
operator action.

## Apply order

Order matters. `004_retention.sql` deletes data and must not run before the
continuous aggregates from `002` have actually materialised.

```bash
psql "$HISTORIAN_DSN" -v ON_ERROR_STOP=1 -f 001_schema.sql
psql "$HISTORIAN_DSN" -v ON_ERROR_STOP=1 -f 002_continuous_aggregates.sql
psql "$HISTORIAN_DSN" -v ON_ERROR_STOP=1 -f 003_compression.sql
psql "$HISTORIAN_DSN" -v ON_ERROR_STOP=1 -f 005_event_log.sql
psql "$HISTORIAN_DSN" -v ON_ERROR_STOP=1 -f 006_roles.sql
# Only after verifying the aggregates below:
psql "$HISTORIAN_DSN" -v ON_ERROR_STOP=1 -f 004_retention.sql
```

Every file is idempotent and safe to re-run.

## Before applying 004 (retention)

`004` starts dropping raw chunks older than 90 days. Confirm the aggregates are
populated first:

```sql
-- Should return a recent bucket, not NULL.
SELECT max(bucket) FROM tag_sample_1m;
SELECT max(bucket) FROM tag_sample_1h;

-- Continuous aggregate jobs should show recent successful runs.
SELECT job_id, last_run_started_at, last_successful_finish, last_run_status
FROM timescaledb_information.job_stats
WHERE hypertable_name IN ('tag_sample', 'tag_sample_1m');
```

The aggregates are created `WITH NO DATA`, so on a database with existing
history you must backfill once before the policy keeps them current:

```sql
CALL refresh_continuous_aggregate('tag_sample_1m', NULL, NULL);
CALL refresh_continuous_aggregate('tag_sample_1h', NULL, NULL);
```

On a large backlog this is slow and I/O heavy. Run it in a maintenance window.

## Which table should a query read?

| Time range        | Read from       | Why                           |
| ----------------- | --------------- | ----------------------------- |
| < 7 days          | `tag_sample`    | Full resolution, uncompressed |
| 7–90 days         | `tag_sample`    | Full resolution, compressed   |
| 90 days – 2 years | `tag_sample_1m` | Raw is gone                   |
| > 2 years         | `tag_sample_1h` | 1-minute rollup is gone       |

Dashboards must select the right source for the selected range. A panel pinned
to `tag_sample` silently returns nothing past 90 days, which reads as "the plant
was off" rather than "you are querying the wrong table".

## Verifying compression

```sql
SELECT
    hypertable_name,
    pg_size_pretty(before_compression_total_bytes) AS before,
    pg_size_pretty(after_compression_total_bytes)  AS after,
    round(
        before_compression_total_bytes::numeric
        / NULLIF(after_compression_total_bytes, 0), 1
    ) AS ratio
FROM hypertable_compression_stats('tag_sample');
```

Expect 10–20x on float process data. If the ratio is closer to 2–3x, check that
`compress_segmentby = 'tag_id'` actually applied — that setting is the single
biggest determinant of the outcome.

## Rollback

Retention and compression policies can be removed without data loss:

```sql
SELECT remove_retention_policy('tag_sample');
SELECT remove_retention_policy('tag_sample_1m');
SELECT remove_compression_policy('tag_sample');
```

Dropping the aggregates and hypertable **is** data loss:

```sql
DROP MATERIALIZED VIEW IF EXISTS tag_sample_1h;
DROP MATERIALIZED VIEW IF EXISTS tag_sample_1m;
DROP TABLE IF EXISTS tag_sample;
```

To disable forwarding entirely without touching the historian, set
`P1AM_TIMESCALE_ENABLED=false` on the control node and restart. SQLite remains
the local source of truth throughout, so this is always a safe fallback.

## Version requirements

- PostgreSQL 14+
- TimescaleDB 2.9+ (hierarchical continuous aggregates)

Compression and continuous aggregates are Timescale License (TSL) features —
free to self-host, source-available rather than OSI-open. See the ADR in
`docs/` for the licensing discussion.
