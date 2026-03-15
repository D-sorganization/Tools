# Testing Improvement Action Plan

Date: 2026-03-15
Repository: Tools
Status: Planned

## Objective

Make `Tools` the reliable provider of shared testing contracts for downstream consumers. The repository should prove that exported Python and Rust modules are correct in isolation and when consumed by sibling repositories.

## Key Problems

1. The main Python CI job runs changed tests plus a tiny smoke subset, which is insufficient for provider-side regression protection.
2. Provider-side package contracts for `upstream_drift_tools`, `signal_toolkit`, `model_generation`, `theme`, and related shared modules are only partially enforced.
3. Cross-repo integration is strong for Rust but weak for shared Python packages.
4. Optional-dependency skips are legitimate in some areas, but they currently leave important exported boundaries under-protected.

## Desired End State

1. Required CI proves provider-side package exports work.
2. Required CI proves at least one downstream Python consumer can install and import the shared packages correctly.
3. Coverage policy is tied to shared-module risk, not just changed-file smoke coverage.
4. Downstream breakages are caught in `Tools` before they land elsewhere.

## Workstreams

### 1. Provider Contract Suite

- Add a required `shared-provider-contracts` test target for exported Python packages.
- Cover importability, minimal runtime behavior, serialization boundaries, and backwards-compatible import paths.
- Ensure failures cannot be hidden behind optional skips for core shared packages.

### 2. Python Downstream Integration

- Add a required cross-repo Python workflow similar to the existing Rust downstream integration workflow.
- Validate at least `Gasification_Model` and `UpstreamDrift` against the current `Tools` branch.
- Fail when consumer bootstrap, editable install, or import contracts break.

### 3. Coverage and Selection Policy

- Tighten the Python CI gate so shared-package changes trigger the full relevant contract suite.
- Keep changed-test optimization only for low-risk areas.
- Define explicit minimum test sets for high-risk shared-package directories.

### 4. Skip/XFail Debt Reduction

- Audit shared-package tests that currently skip due to missing optional dependencies.
- Rework them into deterministic unit tests where possible.
- Isolate truly optional stacks into separate non-blocking jobs.

## Verification Criteria

1. A failing shared-package import in a downstream consumer fails `Tools` CI.
2. Changes under `src/shared/python` or exported package roots run required contract tests automatically.
3. Core shared-package contract tests no longer pass solely due to optional dependency skips.
4. Python and Rust cross-repo integration are both enforced.

## GitHub Tracking

- Meta: `#1544` Testing program: harden shared-package provider contracts and downstream integration
- `#1546` Enforce required shared Python provider-contract suite in CI
- `#1547` Add cross-repo Python consumer integration workflow for Gasification_Model and UpstreamDrift
- `#1548` Audit and reduce skip-heavy shared-package tests
