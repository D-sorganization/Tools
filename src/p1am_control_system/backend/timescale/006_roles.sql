-- 006_roles.sql — least-privilege roles
--
-- Two distinct principals. Grafana must never hold write credentials to the
-- plant historian: a compromised or simply misconfigured dashboard should not
-- be able to alter the process record.
--
-- Passwords are NOT set here. Set them out of band so this file stays safe to
-- commit:
--   ALTER ROLE grafana_ro   WITH PASSWORD '...';
--   ALTER ROLE historian_rw WITH PASSWORD '...';

-- --------------------------------------------------------------- read-only ---

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'grafana_ro') THEN
        CREATE ROLE grafana_ro LOGIN;
    END IF;
END
$$;

GRANT CONNECT ON DATABASE CURRENT_CATALOG TO grafana_ro;
GRANT USAGE ON SCHEMA public TO grafana_ro;

GRANT SELECT ON
    tag_sample, tag_sample_1m, tag_sample_1h,
    event_log,
    tag_definition, plant_equipment, plant_unit, plant_area
TO grafana_ro;

-- Cover objects created by later migrations too.
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO grafana_ro;

-- -------------------------------------------------------------- shipper rw ---
-- The control node's shipper. Needs INSERT on samples and events, and needs to
-- register previously unseen tags. It does NOT get UPDATE or DELETE: the
-- historian is append-only from the controller's point of view, so a bug in the
-- shipper cannot rewrite history.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'historian_rw') THEN
        CREATE ROLE historian_rw LOGIN;
    END IF;
END
$$;

GRANT CONNECT ON DATABASE CURRENT_CATALOG TO historian_rw;
GRANT USAGE ON SCHEMA public TO historian_rw;

GRANT INSERT         ON tag_sample, event_log TO historian_rw;
GRANT SELECT, INSERT ON tag_definition        TO historian_rw;
GRANT SELECT         ON plant_area, plant_unit, plant_equipment TO historian_rw;
GRANT USAGE, SELECT  ON SEQUENCE tag_definition_id_seq TO historian_rw;
GRANT USAGE, SELECT  ON SEQUENCE event_log_id_seq      TO historian_rw;

-- Note: tag_definition also needs UPDATE for the shipper's ON CONFLICT DO
-- UPDATE upsert, which is used only to return an existing id on a race. Grant
-- it narrowly to the name column.
GRANT UPDATE (name) ON tag_definition TO historian_rw;
