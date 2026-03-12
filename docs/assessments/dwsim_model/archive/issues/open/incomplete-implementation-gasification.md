---
title: "Critical: Incomplete implementation in gasification flowsheet"
labels: ["incomplete-implementation", "critical"]
---

## Description

The `_configure_reactors` method in `src/dwsim_model/gasification.py` is incomplete. Variables for reactors are prefixed with `_` to suppress linting errors, masking the fact that they are not configured. `To-Do` comments indicate missing logic for setting up conversion and equilibrium reactions, and configuring volumes and lengths.

## Steps to Reproduce

1. Check `src/dwsim_model/gasification.py`.
2. Inspect the `_configure_reactors` method.

## Expected Behavior

The method should correctly configure the reactors rather than leaving `pass` blocks and `To-Do` comments.
