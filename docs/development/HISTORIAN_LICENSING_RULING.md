# Architecture Ruling: SCADA Historian Licensing Scope

**Issue:** [#4951](https://github.com/D-sorganization/Tools/issues/4951)  
**Parent Program:** [Repository_Management#1505](https://github.com/D-sorganization/Repository_Management/issues/1505) (Fleet Readiness Program 2026-Q4, Pillar P4)  
**Status:** Approved & Ratified  
**Date:** 2026-09-21  
**Scope:** `src/p1am_control_system/historian/`, SCADA telemetry, database schemas, and Grafana dashboard assets  
**Blocks / Unblocks:** Unblocks Historian H3/H6 re-land (Issues #4046, #4049, #4055)  

---

## 1. Context & Problem Statement

The SCADA Truth Audit (#4912, PR #4947) classified 94 of the 111 recoverable telemetry files from closed carrier PRs as **needs-owner**, citing licensing obligations as one of the three primary architectural gates. The recovered schema confirms verbatim usage of **TimescaleDB** features, and operational visualization dashboards target **Grafana**.

1. **Grafana Licensing (AGPLv3):** Since Grafana 8.0, the core application is licensed under the GNU Affero General Public License v3 (AGPLv3). While distributing dashboard definition JSON files is unproblematic, bundling, embedding, hosting modified Grafana runtimes, or offering Grafana over network services triggers AGPL source-disclosure obligations.
2. **TimescaleDB Licensing (Apache-2 vs. TSL):** TimescaleDB splits its codebase between pure Apache-2 (the foundational hypertable storage) and the proprietary **Timescale License (TSL)**. Advanced operational features critical to the recovered schema—including columnar compression, hypertable retention policies, and continuous aggregates—are governed by TSL. TSL strictly forbids offering the software as a managed database service or DBaaS to third parties.

The Tools repository's General Availability (GA) product definition (RM #1505 decision 3) centers on a distributable Rate-of-Closure bundle (standalone Python wheel, desktop executable, and browser companion). A clear licensing boundary is required to determine whether the SCADA historian can leverage TSL and AGPL assets without conflicting with commercial distribution models.

---

## 2. Decision & Formal Ruling

**Ruling: Option 1 — Internal-Only Deployment.**

The SCADA Historian platform is formally designated and ratified as an **internal-only plant and engineering telemetry system**:

1. **Private Infrastructure Boundary:** The historian, TimescaleDB service, and Grafana runtime are executed exclusively on company-controlled physical hardware, internal servers, and private automation networks (e.g. pilot plants, test benches, and internal engineering workstations).
2. **No External Distribution:** The historian runtime, TimescaleDB engine, and Grafana server are **never** bundled, redistributed, or shipped as part of consumer or third-party client deliverables (such as the Rate-of-Closure desktop application, standalone wheels, or client installers).
3. **No Commercial Database Service (DBaaS):** Telemetry persistence and queries are internal operational tools; the system is never offered as a managed database service, hosting platform, or multi-tenant cloud service to third parties.

---

## 3. Rationale & Licensing Compliance Analysis

Under Option 1 (Internal-Only Deployment):

- **TimescaleDB (TSL Compliance):** Section 2.2 of the Timescale License restricts offering TimescaleDB functionality as a commercial service to third parties (managed service / DBaaS). Internal operation on our own hardware for telemetry capture, continuous aggregates, and internal monitoring is explicitly permitted. The project may utilize full TimescaleDB capabilities—including columnar compression, continuous aggregates, hypertable rollups, and chunk retention policies.
- **Grafana (AGPLv3 Compliance):** The AGPLv3 copyleft provisions trigger upon conveyance (distribution) or network interaction with modified software offered to remote users. Running unmodified Grafana instances internally behind company access controls to visualize plant data does not trigger external source disclosure obligations. Exported dashboard JSONs are configuration artifacts and remain unencumbered.
- **Elimination of Schema Fragmentation:** Choosing Option 1 avoids having to rewrite the recovered H3/H6 schema to strip compression and continuous aggregates (which would have severely degraded telemetry write throughput and query latency on high-frequency plant data).

---

## 4. Architectural Invariants & Boundary Rules

To ensure ongoing compliance, the following architectural invariants are strictly enforced:

1. **Distribution Isolation:**
   - No Python packaging profile (e.g. `ud-tools` wheel, `rate_of_closure` distribution) or Docker image targeted for customer delivery shall include, depend upon, or vendor `p1am_control_system/historian` database migration scripts or Grafana binaries.
   - Historian deployments remain confined to internal compose/k8s topologies (`docker-compose.historian.yml` / internal plant manifests).
2. **Future External Distribution Protocol:**
   - If external distribution or on-premises deployment to third-party operators is ever proposed in a future product milestone:
     - Grafana must remain an external dependency installed and licensed independently by the customer/operator; our repository shall only provide unencumbered template definitions.
     - TimescaleDB features must be restricted to Apache-2 core primitives (or alternative permissive stores such as standard PostgreSQL partitions or DuckDB/ClickHouse), with continuous aggregates replaced by application-level rollups.
3. **Artifact Hygiene:**
   - Database connection credentials, production connection URIs, and live plant tags must strictly adhere to `.env` / secret management guidelines; no plant credentials shall ever be committed.

---

## 5. Implementation & Next Actions

1. **Unblock H3/H6 Re-Land:**
   - Unblocks issues [#4046](https://github.com/D-sorganization/Tools/issues/4046), [#4049](https://github.com/D-sorganization/Tools/issues/4049), and [#4055](https://github.com/D-sorganization/Tools/issues/4055).
   - Re-land recovered telemetry schema incorporating continuous aggregates and hypertable retention policies under internal qualification.
2. **Audit Tracking:**
   - Resolves and closes Issue [#4951](https://github.com/D-sorganization/Tools/issues/4951).
