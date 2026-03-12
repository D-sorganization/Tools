---
title: "CRITICAL: Silent failures and incomplete implementations in DWSIM reactor configuration"
labels: ["incomplete-implementation", "critical"]
assignees: []
---

## Description

A code quality review of recent Git history has identified critical incomplete implementations and silent failures within the DWSIM reactor configuration logic, specifically within `src/dwsim_model/chemistry/reactions.py`. These omissions prevent the proper functioning of the simulation and act as a form of CI/CD gaming since failures are not properly bubbled up.

### Details

The module `src/dwsim_model/chemistry/reactions.py` heavily relies on `try: ... except AttributeError: pass` blocks to bypass instances where DWSIM Automation API property setters are not available.

Examples include:

- `configure_pem`: Swallowing errors when attempting to set `ReactorOperationMode` to 0 (Isothermal).
- `configure_trc`: Swallowing errors when attempting to set Arrhenius parameters (`PreExponentialFactor`, `ActivationEnergy`, `ReactionOrder`).

By failing silently rather than explicitly raising `NotImplementedError`, the true state of the codebase is obscured, allowing tests to pass without correctly configuring the reactors.

## Action Required

1. **Address Silent Failures:** Replace `pass` statements in `except AttributeError:` blocks within `src/dwsim_model/chemistry/reactions.py` with appropriate error handling.
2. If the API implementation is incomplete, explicitly raise a `NotImplementedError` so that the `GasificationFlowsheet._configure_reactors` method can catch it and gracefully log a warning without crashing the entire flowsheet build (as per the established project guidelines).
