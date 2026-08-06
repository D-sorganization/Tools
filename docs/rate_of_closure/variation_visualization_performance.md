# Variation Visualization Performance Contract

The visualization layer prepares bounded plot data; it does not rerun swing or
flight physics. Performance tests use 500 trials, 240 common-time samples, one
registered 3D point, and explicit valid-sample masks.

## Local Reference Measurement

Measured on Windows with Python 3.11.9 and Node/Vitest on 2026-08-05:

| Runtime | Dataset | Preparation Time | Traced Peak Memory | Gate |
|---|---:|---:|---:|---:|
| Python covariance/RMS/eigenpairs | 500 x 240 x 3 | approximately 0.016 s | 8.4 MiB | less than 5 s and 100 MB |
| TypeScript RMS/principal spread | 500 x 240 x 3 | approximately 42 ms | not exposed by Vitest | less than 2,000 ms |

The Python source-position array is 2.75 MiB. Peak memory uses Python's
`tracemalloc`; it does not claim to include every native NumPy allocator or GPU
buffer. Arc rendering is independently
bounded to 200,000 vertices with deterministic sample-index decimation. Raw
trial rows remain available through JSON/CSV exports and are never discarded
from the canonical result.

Run the checked gates with:

```powershell
pytest -q tests/rate_of_closure/test_variation_geometry_properties.py
cd src/rate_of_closure/web
npm test -- --run src/model/variationGeometry.performance.test.ts
```

These are regression budgets, not hardware-independent benchmark claims.
Render latency and browser memory must also be recorded in release evidence on
the target deployment hardware.
