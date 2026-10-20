# Grafana Provisioning-as-Code — Level 3 Plant Historian

Grafana dashboard and datasource provisioning for the Level 3 plant historian (Tools #4046, H6 #5545).

## 🛡️ Safety & Non-Authoritative Invariant

The plant historian and its Grafana visualization layer are **strictly non-authoritative**:

- Dashboards and queries are passive observers of process history.
- The historian and Grafana never block, stall, or delay the real-time control loop or safety-critical de-energizing commands.
- Operators must not rely on Grafana dashboard updates for emergency de-energization or machine-safety interlocks; those functions are implemented autonomously in local controller hardware.

## ⚖️ Licensing Standard

- Grafana is licensed under AGPLv3.
- Per the licensing ruling in `docs/development/HISTORIAN_LICENSING_RULING.md` (#4951), this deployment is strictly **internal-only**.
- Dashboards and provisioning manifests are plain declarative JSON and YAML configurations.
- Nothing in this directory may be bundled or distributed into external/client deliverables.
- No Grafana binaries, container tarballs, or third-party executable images are vendored in this repository.

## 📁 Directory Structure

```
deploy/historian/grafana/
├── README.md                                  # This document
├── provisioning/
│   ├── datasources/
│   │   └── timescale.yaml                     # TimescaleDB / PostgreSQL datasource
│   └── dashboards/
│       └── dashboards.yaml                    # File-based dashboard provider definition
└── dashboards/
    └── plant-overview.json                    # Baseline plant telemetry overview (tag_sample_1m)
```

## ⚙️ Environment Configuration

All credentials and endpoints use environment variable interpolation (no plaintext passwords committed):

| Variable             | Default            | Purpose                                        |
| -------------------- | ------------------ | ---------------------------------------------- |
| `HISTORIAN_HOST`     | `timescale`        | Hostname of the TimescaleDB instance           |
| `HISTORIAN_PORT`     | `5432`             | TimescaleDB port                               |
| `HISTORIAN_DB`       | `historian`        | Target database name                           |
| `HISTORIAN_USER`     | `historian_reader` | Read-only database role for Grafana            |
| `HISTORIAN_PASSWORD` | _(required)_       | Password for `HISTORIAN_USER`                  |
| `HISTORIAN_SSLMODE`  | `disable`          | SSL mode (`disable`, `require`, `verify-full`) |

## 📊 Dashboards

### Plant Overview (`uid: plant-overview`)

Visualizes process tag telemetry against the H3 continuous aggregate (`tag_sample_1m`):

- **Process Tag Trends**: 1-minute averaged time series across configured tags.
- **Excursion Envelope**: Minimum and maximum envelope bands preserving transient spikes that an average alone would mask.
- **Tag Statistics & Sample Counts**: Summary metrics and aggregate sample counts per tag.
