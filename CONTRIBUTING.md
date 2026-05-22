# Contributing Guide

Welcome to the Tools monorepo. This guide covers development setup, coding
standards, the testing pipeline, and the pull-request process.

> **Authoritative sources:** [AGENTS.md](AGENTS.md) for agent roles and
> git-workflow conventions; [CLAUDE.md](CLAUDE.md) for CI requirements and
> cross-repo coordination rules.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Setup](#development-setup)
3. [Project Layout](#project-layout)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Running the Quality Gate Locally](#running-the-quality-gate-locally)
7. [Pull Request Process](#pull-request-process)
8. [Security: Secrets & Credentials](#security-secrets--credentials)
9. [Breaking Changes & Cross-Repo Coordination](#breaking-changes--cross-repo-coordination)
10. [Security Reporting](#security-reporting)

---

## Prerequisites

| Tool   | Minimum version | Notes                    |
| ------ | --------------- | ------------------------ |
| Python | 3.11            | 3.12 recommended         |
| Git    | 2.38            | LFS support required     |
| `make` | any             | optional; see `Makefile` |

---

## Development Setup

```bash
# 1. Clone
git clone https://github.com/D-sorganization/Tools.git
cd Tools

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# OR: .venv\Scripts\activate    # Windows

# 3. Install all dependencies + editable package
pip install -r requirements.txt
pip install -e ".[all,dev]"

# 4. (Optional) Run the guided dev-setup helper
python setup_dev.py

# 5. Verify the installation
python -m pytest tests/ -m unit -q --tb=short
```

After setup, `python UnifiedToolsLauncher.py` opens the GUI launcher for all
tools.

---

## Project Layout

```
src/
  shared/python/              # Shared libraries (signal_toolkit, calculators, …)
  data_processing/            # Data Processor tool
  document_processing/        # PDF Renamer tool
  <tool_name>/                # Other standalone tools
tests/                        # Pytest suite mirroring src/
docs/                         # Architecture docs, assessments, tutorials
```

See `docs/ARCHITECTURE_OVERVIEW.md` for a full map and `docs/TOOL_STRUCTURE.md`
for the per-tool layout convention.

---

## Coding Standards

All standards are **enforced by CI** — a PR cannot merge if any check fails.

| Rule                                | Command                            | Rationale                                                  |
| ----------------------------------- | ---------------------------------- | ---------------------------------------------------------- |
| Format: Ruff (88 chars)             | `ruff format .`                    | Single canonical formatter                                 |
| Lint: Ruff                          | `ruff check .`                     | Replaces flake8 + isort                                    |
| Type hints on new code              | `mypy --config-file mypy.ini src/` | Advisory; see `mypy.ini`                                   |
| No `print()` in `src/`              | CI grep                            | Use `logging` instead                                      |
| No `TODO`/`FIXME` without issue ref | CI grep                            | Traceability                                               |
| DbC on public APIs                  | —                                  | `TypeError` for type errors, `ValueError` for range errors |

**Commit messages** follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(signal_toolkit): add Butterworth high-pass filter
fix(thermo): normalise zero-sum composition before MW calculation
docs(contributing): expand PR process section
```

---

## Testing

The project uses **pytest** with 13 custom markers. The most important ones:

| Marker        | When to use                                                   |
| ------------- | ------------------------------------------------------------- |
| `unit`        | Fast, isolated, no I/O                                        |
| `integration` | Cross-module or file-system I/O                               |
| `contract`    | Guards the public API surface that downstream repos depend on |
| `e2e`         | Full pipeline tests                                           |
| `slow`        | Tests that take > 5 s                                         |

**Rule:** every new public function needs at least one `unit` test. If it is
part of the shared API consumed by UpstreamDrift or Gasification_Model, add a
`contract` test too.

```bash
# Run unit tests only (fast)
python -m pytest -m unit -q

# Run unit + integration (pre-push sanity check)
python -m pytest -m "unit or integration" -q --timeout=60

# Run contract tests (guards downstream consumers)
python -m pytest -m contract -q

# Full suite
python -m pytest -n auto --timeout=60
```

Coverage minimum is **40 %** (enforced by CI; see `pyproject.toml`
`[tool.coverage.report]`). Do not weaken existing tests or add `@pytest.mark.skip`
to pass CI — fix the underlying issue.

---

## Running the Quality Gate Locally

Run these before pushing to catch CI failures early:

```bash
# 1. Format
python -m ruff format .

# 2. Lint (auto-fix safe issues)
python -m ruff check . --fix

# 3. Type check (advisory)
python -m mypy src/ --config-file mypy.ini

# 4. Tests
python -m pytest tests/ -m "unit or integration" -q --timeout=60 --tb=short

# 5. Coverage check
python -m pytest tests/ --cov=src --cov-report=term-missing -q
```

All five must pass (mypy output is advisory but should not introduce new errors
on touched files).

---

## Pull Request Process

1. **Branch** from `main`:

   ```bash
   git checkout main && git pull
   git checkout -b fix/my-change
   ```

2. **Implement** the smallest correct change. Do not refactor unrelated code
   in the same PR.

3. **Add tests** for new paths. Do not modify existing tests to pass — fix the
   implementation.

4. **Run the quality gate** (see above).

5. **Open the PR** targeting `main`:

   ```bash
   gh pr create --title "fix(scope): concise description" \
     --body "Fixes #<issue>"
   ```

6. **CI must go green** before merge. If a check fails:

   - Read the annotation in the GitHub UI.
   - Push a fix commit (do not force-push to `main`).
   - Do not add `# noqa` or `# type: ignore` to silence a real error.

7. **Review:** PRs need at least one approving review from a code owner before
   merge. Draft PRs are welcome for early feedback.

8. **Merge method:** squash-merge. The squash commit message is the PR title
   (Conventional Commits format).

### PR Description Template

```markdown
## Summary

One paragraph: what changed and why.

## Changes

- Added X to Y
- Fixed Z in W

## Verification

- [ ] `ruff format . && ruff check .` — zero violations
- [ ] `python -m pytest -m "unit or integration" -q` — all pass
- [ ] Manual smoke test: …

## Out of scope

- …

Fixes #<issue>
```

---

## Security: Secrets & Credentials

**Never commit secrets, API keys, passwords, tokens, or credentials.**

| Pattern                      | Safe alternative                                  |
| ---------------------------- | ------------------------------------------------- |
| `API_KEY = "abc123"`         | `os.getenv("API_KEY")`                            |
| Credentials in test fixtures | `OWASP-TEST-API-KEY-SERVICE-EXAMPLE`              |
| Secrets in `.env`            | `.env` is in `.gitignore`; provide `.env.example` |
| GUI credentials              | OS keyring via `keyring` library                  |

Scan before committing:

```bash
python -c "
from src.python.src.utils.secrets_scanner import scan_directory, report_findings
print(report_findings(scan_directory('src/')))
"
```

See `SECURITY.md` and `docs/SECRETS_MANAGEMENT.md` for full guidance.

---

## Breaking Changes & Cross-Repo Coordination

This library is consumed by **UpstreamDrift** and **Gasification_Model**.
Any change to a public function's signature, return type, or exception
behaviour is a breaking change.

Breaking changes require:

1. A deprecation path (old name kept with a `@deprecated` decorator, new name
   added alongside it).
2. Simultaneous PRs in the downstream repos updating their call-sites.
3. Those PRs linked in your Tools PR description.

If you are unsure whether a change is breaking, run the contract test suite:

```bash
python -m pytest -m contract -v
```

A failing contract test means a downstream repo would break.

---

## Security Reporting

Do not open a public GitHub issue for a vulnerability. Use the process
documented in `SECURITY.md` (private disclosure via GitHub's security advisory
flow).
