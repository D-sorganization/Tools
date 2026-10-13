# SCADA Historian Operator & Engineer Runbook

Operational and engineering guide for the P1AM SCADA historian, covering the local authoritative SQLite persistence engine, dual-write forwarding to TimescaleDB, verification, backfill from spool, failure modes, and emergency rollback to SQLite-only operation.

Governed by `docs/adr/ADR-0050-plant-historian-timescaledb.md` and licensing ruling `docs/development/HISTORIAN_LICENSING_RULING.md` (#4951).

---

## 1. Safety Rules & Operational Principles

1. **Safety Primacy (Non-Negotiable):**
   The historian must **never block, delay, or fail** the 10 Hz PLC poll loop (`P1AM_POLL_INTERVAL_S`) or any de-energizing/E-stop safety command.
2. **Decoupled Asynchronous Execution:**
   The `HistorianWriter` (`src/p1am_control_system/backend/historian.py`) handles all persistence off the asyncio event loop using worker threads (`asyncio.to_thread`). Submissions via `submit()` are strictly non-blocking.
3. **Local Store as Source of Truth:**
   The local SQLite database (`dcs_scada.db`, configured via `P1AM_DB_PATH` in `src/p1am_control_system/backend/database.py`) is the continuous, authoritative plant record. External forwarding is best-effort and secondary.
4. **Alarm Transition Preservation:**
   Under queue backpressure (`DEFAULT_HISTORIAN_QUEUE_SIZE = 256`), tag samples are resampled and the oldest sample is dropped. However, critical alarm transitions and data-quality events are carried forward onto new incoming records so state changes are never lost.

---

## 2. Configuration & Diagnostics Inventory

### Primary Backend Settings (`src/p1am_control_system/backend/settings.py` & `database.py`)

| Setting / Flag | Default | Description |
| :--- | :--- | :--- |
| `P1AM_DB_PATH` | `dcs_scada.db` | Filesystem path for the primary SQLite database. |
| `P1AM_CAPTURE_INTERVAL_S` | `5.0` | Minimum seconds between historian sample writes (0 = every scan). |
| `P1AM_HISTORIAN_MAX_BYTES` | `1073741824` | Byte-cap ceiling (1 GB) before retention pruning triggers in `src/p1am_control_system/backend/data_capture.py`. |
| `P1AM_HISTORIAN_RETENTION_INTERVAL_S` | `300.0` | Interval in seconds between retention cleanup sweeps. |
| `P1AM_SQLITE_SYNCHRONOUS` | `NORMAL` | SQLite synchronous pragma (`NORMAL`, `FULL`, `OFF`). |
| `P1AM_POLL_INTERVAL_S` | `0.1` | Fixed 10 Hz PLC control scan period. |
| `P1AM_REQUIRE_READ_AUTH` | `True` | Authorization gate protecting telemetry and trend query endpoints. |

### Diagnostic Metrics & Loop Counters (`src/p1am_control_system/backend/performance_models.py` & `poll_runtime.py`)

- `historian_write_failures`: Number of batched historian commits abandoned after exhausting retries.
- `historian_samples_dropped`: Total count of tag samples discarded due to buffer capacity saturation.
- `scan_overruns`: Number of control scans that missed their monotonic execution deadline.
- `write_failures`: Process-lifetime write error counter on `HistorianWriter`.
- `dropped_samples`: Process-lifetime dropped sample counter on `HistorianWriter`.
- `rows_written`: Total count of individual tag rows committed to storage.
- `queue_depth`: Current occupancy of the in-memory write queue.

---

## 3. Standard Operating Procedures

### Procedure A: Enabling Dual-Write Forwarding

Dual-write forwarding replicates live tag samples from the local control node to an external TimescaleDB instance for long-term retention and Grafana dashboards.

1. **Verify Licensing Scope:**
   Confirm deployment is internal-only per `docs/development/HISTORIAN_LICENSING_RULING.md`. Do not deploy to external customer sites.
2. **Configure Forwarding Environment:**
   Edit the systemd environment file (`/etc/default/p1am-backend` or compose `.env`) on the control node:
   ```bash
   # Enable external historian shipper
   P1AM_TIMESCALE_FORWARDING=true
   P1AM_TIMESCALE_TARGET_DSN="postgresql://historian_rw:${HISTORIAN_PASSWORD}@historian-host:5432/plant_history"
   ```
3. **Restart the Backend Service:**
   ```bash
   sudo systemctl restart p1am-backend
   ```
4. **Initial Boot Verification:**
   Inspect service logs to confirm that the shipper initialized without errors:
   ```bash
   journalctl -u p1am-backend -n 50 --no-pager
   ```

---

### Procedure B: Verifying Historian Health

Perform these checks immediately after enabling dual-write or during routine audits:

1. **Inspect Loop Diagnostics via API:**
   Request the performance endpoint:
   ```bash
   curl -s http://localhost:8000/api/performance | python3 -m json.tool
   ```
   **Acceptance Criteria:**
   - `historian_write_failures` must be `0` (or not incrementing).
   - `historian_samples_dropped` must be `0`.
   - `scan_overruns` must remain steady without unexpected spikes.

2. **Verify Local SQLite Persistence:**
   Confirm new samples are writing to `TagLog` (`src/p1am_control_system/backend/models.py`):
   ```bash
   sqlite3 dcs_scada.db "SELECT count(*), max(timestamp) FROM taglog;"
   ```

3. **Verify Remote TimescaleDB Ingestion:**
   On the historian host:
   ```bash
   psql "$HISTORIAN_DSN" -c "SELECT count(*), max(ts) FROM tag_sample;"
   ```
   Confirm that `max(ts)` tracks within 5–10 seconds of current UTC time.

4. **Verify Grafana Dashboard:**
   Log into Grafana (`http://historian-host:3000`). Open the **Plant Telemetry** dashboard. Verify that live curves render continuously without synthetic interpolation across gaps.

---

### Procedure C: Backfill from Spool & SQLite

Because live forwarding operates on an at-most-once in-memory queue to guarantee zero stall on the control loop, any network partition or remote database downtime creates a gap in TimescaleDB. The missing data is preserved in the local SQLite database.

1. **Identify the Outage Window:**
   Query TimescaleDB to determine the start and end timestamps of the gap:
   ```sql
   SELECT max(ts) AS last_successful_ts FROM tag_sample;
   ```
2. **Execute Safe Export from SQLite Spool:**
   Extract the missing rows from `TagLog` without locking the active database by using SQLite WAL read mode:
   ```bash
   sqlite3 -header -csv dcs_scada.db \
     "SELECT tag_name, value, timestamp, quality FROM taglog WHERE timestamp > 'YYYY-MM-DD HH:MM:SS';" \
     > /tmp/historian_backfill.csv
   ```
3. **Stream Backfill into TimescaleDB:**
   Use the ingestion utility or Postgres `\copy` with throttled chunks to prevent I/O spikes:
   ```bash
   psql "$HISTORIAN_DSN" -c "\copy tag_sample(tag_name, val, ts, quality) FROM '/tmp/historian_backfill.csv' WITH (FORMAT csv, HEADER true);"
   ```
4. **Trigger Continuous Aggregate Refresh:**
   Manually refresh the 1-minute and 1-hour rollups over the backfilled range:
   ```sql
   CALL refresh_continuous_aggregate('tag_sample_1m', 'YYYY-MM-DD HH:MM:SS', now());
   ```

---

## 4. Failure Modes & Troubleshooting

### 1. Network Partition / Remote Host Down
- **Symptom:** Remote TimescaleDB becomes unreachable.
- **System Response:** The bounded in-memory buffer fills. Once `DEFAULT_HISTORIAN_QUEUE_SIZE` is reached, `HistorianWriter` drops oldest samples while carrying alarm and quality transitions forward. `historian_samples_dropped` increments.
- **Safety Impact:** **Zero.** The PLC poll loop runs unhindered at 10 Hz; local SQLite logging continues without interruption.
- **Recovery:** When network connectivity is restored, live streaming resumes automatically. Run **Procedure C** to backfill the gap from SQLite.

### 2. SQLite Database Lock or Flash Latency
- **Symptom:** SQLite write takes longer than normal due to disk contention or VACUUM.
- **System Response:** The off-loop worker thread absorbs the delay. Batches are retried up to 3 times with exponential backoff.
- **Diagnostics:** If all retries fail, `historian_write_failures` increments and the event is logged.

### 3. Continuous Aggregate Stall Hazard
- **Warning:** In TimescaleDB, if a continuous aggregate rollup job halts while the raw chunk retention policy remains active, raw data will be dropped before summaries are computed, causing irreversible historical data loss.
- **Detection:**
  ```sql
  SELECT job_id, last_run_status, last_successful_finish, total_failures 
  FROM timescaledb_information.job_stats;
  ```
- **Remediation:** If the aggregate job is failing, immediately drop or pause the retention policy:
  ```sql
  SELECT remove_retention_policy('tag_sample');
  ```
  Resolve the aggregate failure, verify successful rollup runs, and re-apply retention.

---

## 5. Rollback Procedure to SQLite-Only

In the event of persistent network instability, remote server corruption, unresolvable driver errors, or planned infrastructure maintenance, follow this explicit procedure to return the plant to **SQLite-only** operation.

### Step 1: Disable External Forwarding
Edit the service environment configuration (`/etc/default/p1am-backend` or `.env`):
```bash
# Disable remote forwarding entirely
P1AM_TIMESCALE_FORWARDING=false
```
Alternatively, remove or comment out `P1AM_TIMESCALE_TARGET_DSN`.

### Step 2: Restart the Backend Service
Apply the configuration change by restarting the control daemon:
```bash
sudo systemctl restart p1am-backend
```

### Step 3: Validate SQLite-Only Operation
1. Confirm that the service started cleanly and the remote forwarder task is inactive:
   ```bash
   journalctl -u p1am-backend -n 30 --no-pager
   ```
2. Verify that `historian_write_failures` does not increment:
   ```bash
   curl -s http://localhost:8000/api/performance | python3 -m json.tool
   ```
3. Verify local telemetry persistence continues into `TagLog`:
   ```bash
   sqlite3 dcs_scada.db "SELECT count(*), max(timestamp) FROM taglog;"
   ```
4. Confirm loop scan stability: `scan_overruns` remains at normal baseline and scan interval stays at 100 ms (`P1AM_POLL_INTERVAL_S = 0.1`).

### Step 4: Decommission Remote Services Safely
With forwarding disabled, remote TimescaleDB and Grafana containers on the historian host can be stopped, backed up, or re-initialized with zero impact on the running plant:
```bash
# On the historian host
docker compose down
```

### Step 5: Data Integrity Guarantee
Because SQLite has recorded all tag samples and alarms locally throughout the deployment, **zero data is lost** during or after rollback.
