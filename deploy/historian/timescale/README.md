# TimescaleDB Plant Historian Schema Migrations

Versioned SQL migrations for the Level 3 Plant Historian (Tools #4046, #5542).

**These migrations are not applied automatically at application startup.**
Schema changes on a plant historian are an intentional operator action.

## Licensing Scope

Per the licensing ruling in `docs/development/HISTORIAN_LICENSING_RULING.md` (#4951):
- Continuous aggregates (`timescaledb.continuous`), compression policies, and retention policies are Timescale License (TSL) features.
- These features are permitted for internal plant deployment.
- Nothing in this directory may be bundled into distributed Rate-of-Closure deliverables.

## Safety & Non-Blocking Guarantee

The historian is strictly non-authoritative. The shipper and write backends must never block, delay, or fail the control poll loop or de-energizing safety commands.

## Apply Order

Order matters. `004_retention.sql` drops raw chunks older than 90 days and must not run before the continuous aggregates from `002_continuous_aggregates.sql` have been applied.

```bash
psql "$HISTORIAN_DSN" -v ON_ERROR_STOP=1 -f 001_hypertable.sql
psql "$HISTORIAN_DSN" -v ON_ERROR_STOP=1 -f 002_continuous_aggregates.sql
psql "$HISTORIAN_DSN" -v ON_ERROR_STOP=1 -f 003_compression.sql
# Apply retention once aggregates are verified:
psql "$HISTORIAN_DSN" -v ON_ERROR_STOP=1 -f 004_retention.sql
```

Every file is idempotent (`IF NOT EXISTS` / `if_not_exists => TRUE`) and safe to re-run.

## Verification

### 1. Verify Hypertable
```sql
SELECT hypertable_name, num_chunks
FROM timescaledb_information.hypertables
WHERE hypertable_name = 'tag_sample';
```

### 2. Verify Continuous Aggregate
```sql
SELECT view_name, materialized_only
FROM timescaledb_information.continuous_aggregates
WHERE view_name = 'tag_sample_1m';
```

### 3. Verify Compression
```sql
SELECT
    hypertable_name,
    before_compression_total_bytes,
    after_compression_total_bytes
FROM hypertable_compression_stats('tag_sample');
```

## Rollback Procedures

Policies can be removed without data loss:

```sql
-- Remove retention policy:
SELECT remove_retention_policy('tag_sample', if_exists => TRUE);

-- Remove compression policy:
SELECT remove_compression_policy('tag_sample', if_exists => TRUE);
ALTER TABLE tag_sample SET (timescaledb.compress = false);

-- Remove continuous aggregate policy:
SELECT remove_continuous_aggregate_policy('tag_sample_1m', if_exists => TRUE);
DROP MATERIALIZED VIEW IF EXISTS tag_sample_1m;
```

Dropping tables completely (destroys data):
```sql
DROP TABLE IF EXISTS tag_sample CASCADE;
DROP TABLE IF EXISTS tag_definition CASCADE;
DROP TABLE IF EXISTS plant_equipment CASCADE;
DROP TABLE IF EXISTS plant_unit CASCADE;
DROP TABLE IF EXISTS plant_area CASCADE;
```
