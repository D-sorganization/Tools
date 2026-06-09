# Deployment

Tools is a Python library — it does not ship as a standalone service.
However, containerised images are provided for three use-cases:

| Image          | Dockerfile        | Purpose                                                  |
| -------------- | ----------------- | -------------------------------------------------------- |
| `tools:latest` | `Dockerfile`      | Run any Tools-based script in a reproducible environment |
| `tools:dev`    | `Dockerfile.dev`  | Live development with auto-reload and dev extras         |
| `tools:prod`   | `Dockerfile.prod` | Minimal production image for web application deployments |

---

## Quick start

### Run a script

```bash
docker build -t tools:latest .
docker run --rm \
  -v "$(pwd)/my_script.py:/workspace/my_script.py" \
  tools:latest python3 my_script.py
```

### Interactive shell

```bash
docker run --rm -it tools:latest python3
```

### Local development (all services)

```bash
docker-compose up
```

The `docker-compose.yml` starts the Tools Flask web application on port 5000
with live code reload, plus optional PostgreSQL and Redis sidecars.

---

## Image details

### `Dockerfile` (generic runner)

- Base: `python:3.11-slim`
- Installs `.[all]` extras (no dev/test tools — keep the image lean)
- Runs as non-root user `appuser`
- Default `CMD` is `python3` (interactive); override at runtime:

  ```bash
  docker run --rm tools:latest python3 -m src.signal_processing.my_module
  ```

### `Dockerfile.dev` (development)

- Base: `python:3.11-slim`
- Installs `.[all,dev]` in editable mode
- Mounts the full project directory for live editing
- Default `CMD` is `python3`

### `Dockerfile.prod` (production)

- Multi-stage build; final image contains only runtime wheels
- Optimised for minimal attack surface and image size

---

## Environment variables

| Variable          | Default      | Description                                                       |
| ----------------- | ------------ | ----------------------------------------------------------------- |
| `FLASK_APP`       | —            | Flask application path, e.g. `web_applications.calculator.webapp` |
| `FLASK_ENV`       | `production` | `development` enables auto-reload                                 |
| `WEB_CONCURRENCY` | `1`          | Number of Gunicorn/Flask workers                                  |
| `SECRET_KEY`      | —            | Required in production; set a strong random value                 |

---

## `.dockerignore`

The `.dockerignore` at the project root excludes `.git`, test fixtures,
documentation, IDE config, and large media files to keep build context small
and avoid leaking developer credentials into the image.

---

## CI/CD

Docker images are built and pushed in the GitHub Actions pipeline defined in
`.github/workflows/`. The generic `Dockerfile` is built on every commit to
`main`; `Dockerfile.prod` is built and tagged on version tags (`v*`).

## Package publishing (PyPI + NPM)

Public package publishing is handled by `.github/workflows/publish-artifacts.yml`.
Both publish jobs run **only** on a GitHub `release` event or a manual
`workflow_dispatch` with `dry_run=false`, and both are gated behind a protected
GitHub deployment environment that requires reviewer approval before the publish
step runs:

| Target | Job            | Protected environment | Credential               |
| ------ | -------------- | --------------------- | ------------------------ |
| PyPI   | `publish-pypi` | `pypi`                | `PYPI_API_TOKEN`         |
| NPM    | `publish-npm`  | `npm`                 | `NODE_AUTH_TOKEN` / OIDC |

A manual `workflow_dispatch` with `dry_run=false` therefore cannot publish to
**either** registry without an environment approval, removing the previous
asymmetry where NPM could be published without the review gate that protected
PyPI.

### Rollback / unpublish constraints

Both registries treat a published version as effectively immutable; the safe
recovery path is to publish a fixed higher version, not to overwrite.

- **PyPI:** a released version filename can never be re-uploaded. Deleting a
  release is permanent and does **not** free the version string for reuse. Yank
  a bad release (hides it from resolvers but keeps existing pins working), then
  publish a patched version.
- **NPM:** `npm unpublish` is only permitted within 72 hours of publish and
  only when nothing depends on the version; otherwise use `npm deprecate` and
  publish a patched version. Re-publishing an unpublished version string is
  blocked for 24 hours.

Treat any unintended publish to either registry as an incident: deprecate/yank
the bad version immediately and ship a corrected higher version.
