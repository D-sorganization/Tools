## 2026-04-23 - AnalyticsSuite computePCA Optimization
**Learning:** Chained array methods (`.map`, `.reduce`) and spread operators inside tight mathematical loops like PCA or regression cause severe garbage collection stuttering due to constant reallocation.
**Action:** Replace functional array iteration chains with pre-allocated arrays and single-pass standard `for` loops when writing complex data processing pipelines in JS/TS.
