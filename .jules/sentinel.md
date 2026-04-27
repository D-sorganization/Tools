## 2026-04-06 - [CRITICAL] Disable Insecure Pickle Deserialization
**Vulnerability:** The data processing utilities (`file_utils.py`, `io.py`) supported parsing `.pkl`/`.pickle` files using Pandas' `pd.read_pickle`.
**Learning:** This function internally uses Python's built-in `pickle` module, which exposes a severe CWE-502 vulnerability when processing untrusted files (arbitrary code execution). Because these formats could be submitted via UI or API by external users, retaining support was a major risk.
**Prevention:** Unsafe serialization formats like `pickle` must not be explicitly exposed to user-provided input streams. Stick to safe formats like Parquet, JSON, or CSV.
