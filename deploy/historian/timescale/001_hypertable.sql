-- 001_hypertable.sql — plant historian core schema and sample hypertable
--
-- Architecture: Level 3 Plant Historian (Tools #4046, #5542)
-- Requires: PostgreSQL 14+ with TimescaleDB extension.
--
-- Safety notice: The historian is strictly non-authoritative and must never block,
-- delay, or fail the control poll loop or de-energizing safety commands.
--
-- Licensing notice: Continuous aggregates and compression use Timescale License (TSL)
-- features permitted under the internal-only deployment ruling (#4951). These artifacts
-- must not be bundled into distributed Rate-of-Closure deliverables.
--
-- Idempotent: safe to re-run.
-- Schema changes against a production historian are an operator action, deliberately.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ---------------------------------------------------------------------------
-- Asset hierarchy
--
-- Mirrors SQLModel definitions (PlantArea -> PlantUnit -> PlantEquipment -> TagDefinition).
-- Keeping hierarchy in the same database enables asset-scoped relational queries
-- (e.g. "all temperature tags in Unit R-101") without requiring the client to
-- know every tag name ahead of time.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS plant_area (
    id    SERIAL PRIMARY KEY,
    name  TEXT   NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS plant_unit (
    id       SERIAL PRIMARY KEY,
    name     TEXT    NOT NULL,
    area_id  INTEGER NOT NULL REFERENCES plant_area (id) ON DELETE CASCADE,
    UNIQUE (area_id, name)
);

CREATE TABLE IF NOT EXISTS plant_equipment (
    id       SERIAL PRIMARY KEY,
    name     TEXT    NOT NULL,
    unit_id  INTEGER NOT NULL REFERENCES plant_unit (id) ON DELETE CASCADE,
    UNIQUE (unit_id, name)
);

-- Tag definitions. `name` is the natural key the controller knows (e.g. TAG_0..);
-- `id` is the surrogate integer key the hypertable stores to reduce per-sample identity cost.
-- Nullable engineering metadata ensures unseen tags never block ingestion.
CREATE TABLE IF NOT EXISTS tag_definition (
    id                SERIAL PRIMARY KEY,
    name              TEXT NOT NULL UNIQUE,
    description       TEXT NOT NULL DEFAULT '',
    engineering_units TEXT,
    tag_type          TEXT,
    equipment_id      INTEGER REFERENCES plant_equipment (id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS ix_tag_definition_equipment
    ON tag_definition (equipment_id);

-- ---------------------------------------------------------------------------
-- Sample hypertable
--
-- `quality` is present from initial creation to avoid costly decompression/migration
-- on billions of rows later. Values follow the OPC DA / OPC UA standard convention:
-- 192 = Good, 64 = Uncertain, 0 = Bad.
-- Deliberately append-only with no surrogate primary key to minimize index overhead.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS tag_sample (
    ts       TIMESTAMPTZ      NOT NULL,
    tag_id   INTEGER          NOT NULL REFERENCES tag_definition (id),
    value    DOUBLE PRECISION NOT NULL,
    quality  SMALLINT         NOT NULL DEFAULT 192
);

SELECT create_hypertable(
    'tag_sample',
    'ts',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

-- Dominant read pattern: one tag over a time range, ordered by timestamp descending.
CREATE INDEX IF NOT EXISTS ix_tag_sample_tag_ts
    ON tag_sample (tag_id, ts DESC);

COMMENT ON TABLE tag_sample IS
    'Raw process samples. Retention 90 days; see 004_retention.sql. Long-horizon history '
    'lives in the tag_sample_1m continuous aggregate.';
COMMENT ON COLUMN tag_sample.quality IS
    'OPC style quality code. 192 = Good, 64 = Uncertain, 0 = Bad.';

-- ---------------------------------------------------------------------------
-- Rollback / Down note
-- ---------------------------------------------------------------------------
-- To remove the hypertable and asset hierarchy (WARNING: Destroys sample data):
--   DROP TABLE IF EXISTS tag_sample CASCADE;
--   DROP TABLE IF EXISTS tag_definition CASCADE;
--   DROP TABLE IF EXISTS plant_equipment CASCADE;
--   DROP TABLE IF EXISTS plant_unit CASCADE;
--   DROP TABLE IF EXISTS plant_area CASCADE;
