-- 001_schema.sql — plant historian core schema
--
-- Requires: PostgreSQL 14+ with the timescaledb extension (2.9+ for the
-- hierarchical continuous aggregate created in 002).
--
-- Idempotent: safe to re-run. Apply with the ordering in README.md.
--
-- NOT applied automatically at application startup. Schema changes against a
-- production historian are an operator action, deliberately.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ---------------------------------------------------------------------------
-- Asset hierarchy
--
-- Mirrors the SQLModel definitions in backend/models.py (PlantArea ->
-- PlantUnit -> PlantEquipment -> TagDefinitionDb). Keeping the hierarchy in the
-- same database as the samples is the whole reason this is TimescaleDB rather
-- than a pure metrics store: it lets a query ask "every temperature in R-101"
-- instead of requiring the caller to already know the tag names.
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

-- Tag definitions. `name` is the natural key the controller knows (TAG_0..);
-- `id` is the surrogate the hypertable stores, so a sample costs 4 bytes of
-- identity rather than a repeated string.
--
-- The shipper auto-registers unknown tags with name only. Engineering metadata
-- (description, units, equipment_id) is expected to be filled in afterwards and
-- is therefore all nullable — an unlabelled tag must never block ingest.
CREATE TABLE IF NOT EXISTS tag_definition (
    id            SERIAL PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    description   TEXT NOT NULL DEFAULT '',
    engineering_units TEXT,
    tag_type      TEXT,
    equipment_id  INTEGER REFERENCES plant_equipment (id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS ix_tag_definition_equipment
    ON tag_definition (equipment_id);

-- ---------------------------------------------------------------------------
-- Sample hypertable
--
-- `quality` is present from the first migration even though the P1AM path only
-- ever writes "good" today. Adding a column to a compressed, multi-billion-row
-- hypertable later means decompressing it; a SMALLINT with a default costs
-- nothing now. Values follow the OPC UA convention (192 = Good, 0 = Bad,
-- 64 = Uncertain), which is what any future OPC UA or Sparkplug ingest will
-- already be speaking.
--
-- Deliberately no surrogate primary key: a PK would add a unique index over
-- every row for no benefit, and this table is append-only.
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

-- Serves the dominant read pattern: one tag over a time range, ordered by time.
-- `ts DESC` matches "most recent N samples" and the trend queries Grafana emits.
CREATE INDEX IF NOT EXISTS ix_tag_sample_tag_ts
    ON tag_sample (tag_id, ts DESC);

COMMENT ON TABLE  tag_sample IS
    'Raw process samples. Retention 90 days; see 004_retention.sql. Long-horizon '
    'history lives in the tag_sample_1m / tag_sample_1h continuous aggregates.';
COMMENT ON COLUMN tag_sample.quality IS
    'OPC UA style quality code. 192 = Good, 64 = Uncertain, 0 = Bad.';
