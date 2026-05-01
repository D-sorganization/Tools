# Helm / Kubernetes Deployment Notes

## Tools is a Pure Python Library — No Server Process to Deploy

`Tools` (distributed as the `ud-tools` pip package) is a **shared
engineering library**, not a server or API process.  It has no HTTP
endpoints, no listening sockets, and no long-running daemon to containerise
or schedule.

Kubernetes manifests and Helm charts are therefore **not applicable** to
this repository itself.

---

## How Downstream Repos Deploy Tools Code

Functionality from `Tools` reaches production as part of the downstream
applications that consume it:

| Downstream repo | Deployment pattern |
|---|---|
| **UpstreamDrift** | Docker image built from its own `Dockerfile`; `ud-tools` is installed as a Python dependency during the image build (`pip install ud-tools==<pin>`). |
| **Gasification_Model** | Same pattern — `ud-tools` is a `requirements.txt` / `pyproject.toml` dependency installed at image build time. |

Both downstream repos ship their own Kubernetes manifests and Helm charts.
`Tools` itself appears only as a layer inside those images.

---

## Updating the Library Version in Downstream Deployments

When a new version of `ud-tools` is published:

1. **Pin the new version** in the downstream repo's dependency file
   (`requirements-lock.txt` or `pyproject.toml`).
2. **Run downstream contract tests** (`pytest -m contract`) to verify no
   API surface regressions.  See `tests/integration/test_cross_repo_contracts.py`
   in this repo for the test suite that mirrors those checks.
3. **Rebuild the downstream Docker image** with the updated pin.
4. **Deploy** via the downstream repo's own Helm chart or Kubernetes
   manifests.

---

## Resource Sizing Guidance (for Downstream Operators)

The following guidance applies to containers that **include** `ud-tools`:

| Workload type | Recommended baseline |
|---|---|
| Signal processing (scipy-heavy) | 1 CPU / 512 MiB RAM |
| Process calculators (pressure drop, flare sizing) | 0.5 CPU / 256 MiB RAM |
| URDF / model generation | 1 CPU / 1 GiB RAM (trimesh in-memory mesh ops) |
| Data processor (pandas, large CSV) | 1 CPU / 1–4 GiB RAM (depends on file size) |

These are starting points.  Profile with representative workloads before
setting hard resource limits in production manifests.

---

## Health-Check Considerations

`ud-tools` provides no HTTP health endpoint.  Downstream containers that
expose an HTTP API (e.g. via FastAPI + `calc_backend`) should wire their
own `/health` route.  The `calc_backend` module in this repo includes a
`HealthChecker` class (`src/shared/python/calc_backend/health.py`) that
downstream API servers can reuse.

---

## Secrets and Configuration

`ud-tools` itself reads no secrets and requires no runtime environment
variables.  Configuration for downstream deployments (API keys, database
URLs, feature flags) belongs in the downstream repo's `ConfigMap` /
`Secret` resources, not here.

---

## Further Reading

- `docs/adr/ADR-002-shared-library-module-structure.md` — why Tools is a
  library, not a service.
- `docs/adr/ADR-005-plugin-discovery-vs-registry.md` — how tools are
  registered and discovered at launcher startup (desktop deployment only).
- `DOCKER.md` — Docker development environment for this repo (used for
  contributor workflows, not production deployment).
- `docs/deployment/` — scalability and load-balancing guidance for
  downstream operators.
