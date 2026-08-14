# Plant historian runbook

TimescaleDB + Grafana as a Level 3 information layer above the P1AM control
system. See the epic (#4046) for the architecture and
[ADR-007](../../../../docs/adr/ADR-007-plant-historian-timescaledb.md) for why
this stack was chosen and what would cause us to revisit it.

## What this is and is not

- **Is:** a long-horizon process record, plant analytics, and alarm-performance
  reporting.
- **Is not:** an HMI, a control system, or an operator alarm surface. Grafana is
  read-only and holds read-only database credentials. If Grafana and the HMI
  disagree, **the HMI is authoritative**.

The control node is unaffected by anything here. Its local SQLite historian
remains the source of truth, and forwarding is best-effort — see
"Delivery guarantees" below.

## Topology

```
[Control Pi]                              [Historian host]
  P1AM firmware (interlocks, PID)
  FastAPI backend @ 10 Hz         ──ship──▶  TimescaleDB :5432
  React HMI                                  Grafana     :3000
  SQLite  (local source of truth)
```

Data flows **one way**. Nothing on the historian host initiates a connection
back to the control network. That is the point of the layering: a compromised
Grafana must not be a path to the PLC. Enforce it at the firewall, not by
convention.

Run these on **separate hosts**. TimescaleDB and Grafana on the control Pi will
steal CPU from the 10 Hz scan loop and cause overruns.

## First-time setup

### 1. Historian host

```bash
cd src/p1am_control_system/deploy/historian
cp .env.example .env
# Edit .env — every CHANGE_ME must be replaced.
docker compose up -d
```

### 2. Apply the schema

Migrations are **not** auto-applied. See
[`../../backend/timescale/README.md`](../../backend/timescale/README.md) for the
apply order and the mandatory check before enabling retention.

```bash
cd ../../backend/timescale
# PGPASSWORD rather than a password in the DSN: anything on a command line is
# visible in `ps` and lands in shell history.
read -rs -p "historian_admin password: " PGPASSWORD && export PGPASSWORD
export HISTORIAN_DSN="postgresql://historian_admin@localhost:5432/plant_history"
psql "$HISTORIAN_DSN" -v ON_ERROR_STOP=1 -f 001_schema.sql
psql "$HISTORIAN_DSN" -v ON_ERROR_STOP=1 -f 002_continuous_aggregates.sql
psql "$HISTORIAN_DSN" -v ON_ERROR_STOP=1 -f 003_compression.sql
psql "$HISTORIAN_DSN" -v ON_ERROR_STOP=1 -f 005_event_log.sql
psql "$HISTORIAN_DSN" -v ON_ERROR_STOP=1 -f 006_roles.sql
```

Then set the role passwords to match `.env`:

```sql
ALTER ROLE grafana_ro   WITH PASSWORD '...';   -- GRAFANA_RO_PASSWORD
ALTER ROLE historian_rw WITH PASSWORD '...';   -- used by the Pi
```

Apply `004_retention.sql` only after confirming the aggregates are populating.
It drops raw chunks; running it early loses history permanently.

### 3. Enable forwarding on the control Pi

```bash
export P1AM_TIMESCALE_ENABLED=true
export P1AM_TIMESCALE_DSN="postgresql://historian_rw:PASSWORD@historian-host:5432/plant_history"  # pragma: allowlist secret
```

This one does carry the password inline — it is the single configuration value
the backend reads. Put it in the systemd unit's `EnvironmentFile=` with mode
`0600` and owned by the service user, not in a shell profile. The backend never
logs it in full (see `timescale_writer.redact_dsn`), so it should not appear in
a log bundle; do not paste it into an issue either.

Restart the backend. Startup fails loudly if `ENABLED=true` and the DSN is
empty — a historian everyone believes is recording but is not is worse than one
that is openly off.

Optional tuning (defaults shown):

| Variable                           | Default  | Purpose                                      |
| ---------------------------------- | -------- | -------------------------------------------- |
| `P1AM_TIMESCALE_QUEUE_MAX`         | `100000` | Bounded forward queue; overflow drops oldest |
| `P1AM_TIMESCALE_BATCH_SIZE`        | `1000`   | Samples per round-trip                       |
| `P1AM_TIMESCALE_FLUSH_INTERVAL_S`  | `1.0`    | Max partial-batch latency                    |
| `P1AM_TIMESCALE_CONNECT_TIMEOUT_S` | `5.0`    | Fail-fast connect bound                      |
| `P1AM_TIMESCALE_SHUTDOWN_FLUSH_S`  | `5.0`    | Bound on shutdown flush                      |

### 4. Verify end to end

```bash
# On the Pi — should show connected=true and a climbing shipped_total.
curl -s localhost:8000/api/historian/shipper | python3 -m json.tool

# On the historian host — should return a recent timestamp.
psql "$HISTORIAN_DSN" -c "SELECT max(ts), count(*) FROM tag_sample;"
```

Then open Grafana at `http://historian-host:3000`, folder **Plant**. The
_Historian Health (ingest)_ dashboard should show lag near your capture
interval.

## Delivery guarantees

**At-most-once, deliberately.** The forward queue is in memory; a backend
restart discards whatever had not shipped. SQLite on the Pi holds the
authoritative copy, so a restart loses _forwarding_, never _data_. There is no
automatic backfill from SQLite — do not build anything that assumes
exactly-once.

Under sustained backpressure the queue drops the **oldest** samples. For process
history the newest data is the operationally useful data, and an unbounded queue
on a Pi is an out-of-memory crash of the control node, which is far worse than a
gap in a trend.

## Troubleshooting

### Shipper will not connect

```bash
curl -s localhost:8000/api/historian/shipper | python3 -m json.tool
```

`connected: false` with a rising `consecutive_failures` and a `last_error`:

- `ConnectionRefusedError` — historian container down, or 5432 bound to
  loopback only on the historian host while the Pi is remote. The compose file
  binds `127.0.0.1:5432` by default; expose it on a private interface or VPN,
  never on the plant network.
- `password authentication failed` — role password not set, or `.env` and the
  `ALTER ROLE` diverged.
- `RuntimeError: psycopg is required` — driver not installed on the Pi:
  `pip install 'psycopg[binary]'`.
- `relation "tag_definition" does not exist` — migrations not applied.

### Queue filling / drops climbing

`queue_depth` near `queue_max` with `dropped_total` rising means the shipper
cannot keep up or is disconnected.

1. Check `connected`. A disconnected shipper fills the queue by definition.
2. If connected, the remote is too slow: raise `P1AM_TIMESCALE_BATCH_SIZE`, or
   check historian-host disk I/O.
3. Raising `queue_max` buys time during an outage; it does not fix a sustained
   rate mismatch, and it costs Pi memory.

### Lag climbing while the process runs

Data is not reaching the historian. Trends will have holes. Confirm on the
_Historian Health_ dashboard before interpreting any flat line as a real
measurement.

### Compression not running / poor ratio

```sql
SELECT * FROM timescaledb_information.jobs WHERE proc_name = 'policy_compression';
SELECT * FROM hypertable_compression_stats('tag_sample');
```

A ratio near 2-3x instead of 10-20x almost always means
`compress_segmentby = 'tag_id'` did not apply.

### Continuous aggregate not refreshing

```sql
SELECT job_id, last_run_status, last_successful_finish, total_failures
FROM timescaledb_information.job_stats;
```

**This is the dangerous one.** If the 1-minute aggregate stops refreshing while
the retention policy keeps dropping raw chunks, history is destroyed rather than
downsampled. If aggregates are failing, remove the retention policy until it is
fixed:

```sql
SELECT remove_retention_policy('tag_sample');
```

## Backup

The Grafana volume holds only users and preferences — dashboards live in git.
The historian volume is the plant record.

```bash
# Logical backup (portable, slower)
docker exec plant_historian_db pg_dump -U historian_admin -Fc plant_history \
  > plant_history_$(date +%Y%m%d).dump

# Restore
docker exec -i plant_historian_db pg_restore -U historian_admin \
  -d plant_history --clean --if-exists < plant_history_YYYYMMDD.dump
```

Test a restore before you need one. An untested backup is a hypothesis.

## Rollback

To stop forwarding without touching the historian — one variable and a restart:

```bash
export P1AM_TIMESCALE_ENABLED=false
```

The backend returns to SQLite-only. No code change, no migration, no data loss:
the local historian has been recording the whole time.

To remove the policies without losing data, see the rollback section of
[`../../backend/timescale/README.md`](../../backend/timescale/README.md).

## Security notes

- Grafana holds **read-only** credentials (`grafana_ro`). The shipper role
  (`historian_rw`) has INSERT but no UPDATE or DELETE on samples, so a shipper
  bug cannot rewrite history.
- **Grafana OSS has no per-dashboard RBAC** — that is an Enterprise feature.
  Only org- and folder-level roles exist. Anyone who can log into Grafana can
  see every dashboard in their org. Do not rely on Grafana for access
  segregation between operating areas.
- Change the default admin password on first login.
- Anonymous access is disabled in the compose file. Keep it that way.
- Put Grafana behind a reverse proxy with TLS before exposing it beyond
  loopback.
- The DSN carries a password and is redacted wherever the backend logs it. Do
  not paste an unredacted DSN into an issue or a log bundle.
