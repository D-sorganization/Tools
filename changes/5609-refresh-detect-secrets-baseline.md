---
issue: 5609
summary: "Refresh detect-secrets baseline for SCADA f-matrix test SHAs"
---

- Update `.secrets.baseline` to include the commit SHAs cited as evidence in `tests/scada/test_f_matrix.py` and reconcile removed findings from `docker-compose.yml`.
- Resolves "main is red: Detect Secrets" (#5609).
