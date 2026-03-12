---
title: Improve Test Coverage
labels: jules:assessment,needs-attention
---

Score is 4/10.

Based on our automated assessment, the test coverage (48%) is currently below the acceptable threshold. Test coverage is 48% across the `src/dwsim_model` module, with notably poor test coverage for the standalone models (under 20%). A score below 5/10 requires a GitHub issue for attention.

## Missing tests:

- `src/dwsim_model/standalone/gasifier_model.py` (14% coverage)
- `src/dwsim_model/standalone/pem_model.py` (14% coverage)
- `src/dwsim_model/standalone/trc_model.py` (18% coverage)
- `src/dwsim_model/config_loader.py` (14% coverage)
- `src/dwsim_model/core.py` (39% coverage)
- `src/dwsim_model/gasification.py` (10% coverage)

Improve test coverage across these modules.
