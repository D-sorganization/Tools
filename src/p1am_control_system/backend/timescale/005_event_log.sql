-- 005_event_log.sql — alarm and system events
--
-- Mirrors backend/models.py::EventLog. Separate from tag_sample because events
-- are sparse, textual, and queried by type/severity rather than by tag range —
-- putting them in the sample hypertable would poison its compression.
--
-- This table is what the ISA-18.2 / EEMUA 191 alarm-performance dashboard reads.
-- Without it, that dashboard has no source.

CREATE TABLE IF NOT EXISTS event_log (
    id           BIGSERIAL,
    ts           TIMESTAMPTZ NOT NULL,
    event_type   TEXT        NOT NULL,   -- ALARM | SYSTEM | ACKNOWLEDGE
    description  TEXT        NOT NULL,
    severity     SMALLINT    NOT NULL DEFAULT 0,  -- 0 normal, 1 Hi/Lo, 2 HiHi/LoLo
    tag_id       INTEGER     REFERENCES tag_definition (id) ON DELETE SET NULL,
    PRIMARY KEY (id, ts)
);

SELECT create_hypertable(
    'event_log',
    'ts',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists       => TRUE
);

CREATE INDEX IF NOT EXISTS ix_event_log_type_ts
    ON event_log (event_type, ts DESC);

CREATE INDEX IF NOT EXISTS ix_event_log_severity_ts
    ON event_log (severity, ts DESC);

-- Alarm history is a compliance artefact and is small relative to process
-- samples. No retention policy: keep it all.

COMMENT ON TABLE event_log IS
    'Alarm/system/acknowledge events. Source for the EEMUA 191 alarm '
    'performance dashboard. No retention policy — alarm history is retained '
    'indefinitely as a compliance record.';
