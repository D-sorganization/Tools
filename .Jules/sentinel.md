## 2026-04-06 - [CRITICAL] Disable Insecure Pickle Deserialization
**Vulnerability:** The data processing utilities (`file_utils.py`, `io.py`) supported parsing `.pkl`/`.pickle` files using Pandas' `pd.read_pickle`.
**Learning:** This function internally uses Python's built-in `pickle` module, which exposes a severe CWE-502 vulnerability when processing untrusted files (arbitrary code execution). Because these formats could be submitted via UI or API by external users, retaining support was a major risk.
**Prevention:** Unsafe serialization formats like `pickle` must not be explicitly exposed to user-provided input streams. Stick to safe formats like Parquet, JSON, or CSV.
## 2025-05-24 - CSRF Cookie Parsing Equals Sign Truncation
**Vulnerability:** CSRF token validation failed if the token string contained an equals sign (`=`). The cookie parsing logic `cookie.trim().split('=')` destructured only the first two elements `[key, value]`, inadvertently truncating valid tokens that contain `=`. This effectively blocked legitimate requests under this edge case.
**Learning:** Naive string splitting on `=` is unsafe for HTTP cookies because cookie values are allowed to contain equals signs (e.g., Base64-encoded strings with padding).
**Prevention:** Use `const [key, ...valueParts] = cookie.split('='); const value = valueParts.join('=');` or better, use standard cookie parsing libraries like `cookie` from npm instead of manual string manipulation.
## 2026-04-06 - [HIGH] Fix expression injection vulnerability
**Vulnerability:** The mathematical expression evaluation endpoint lacked comprehensive structural validation for injected code snippets. Users could bypass expression evaluation and trigger code execution or access Python object hierarchies by inputting patterns like `__init__`, `__base__`, `async `, `await `, or basic loop constructs (`try:`, `except:`).
**Learning:** Even when relying on safe parsing frameworks (like `parse_expr()` with allowlists), fundamental structural checking against dangerous keywords is necessary as a first layer of defense to block payload injections before parsing begins.
**Prevention:** Hardened `_validate_security` with a strict denylist covering Python evaluation keywords, object lifecycle hooks, and asynchronous constructs. Always perform early exit sanitization for potentially dangerous inputs.
