## 2024-05-19 - [CRITICAL] Fix command injection in ShellTool
**Vulnerability:** ShellTool's `_is_command_allowed` function used naive string prefix checking (`command.startswith()`) and passed the command to `subprocess.run(..., shell=True)` (implicitly via `["-c", command]`). This allowed command chaining (e.g. `ls; rm -rf /`).
**Learning:** Naive prefix checking is never sufficient for security-sensitive command authorization, particularly when executed using a shell or pseudo-shell. Attackers can trivially bypass the check using command injection techniques like `;`, `&&`, `|`, `&`, and `$()`.
**Prevention:** Always parse command strings strictly using `shlex.split`, verify individual tokens, and explicitly block shell operators to prevent execution of unauthenticated or malicious commands when executing commands via a shell interpreter.
## 2024-05-20 - Prevent Error Information Leakage in Health Checks
**Vulnerability:** API endpoints (`/api/health`, `/api/ready`) caught generic exceptions and returned `str(e)` directly in the JSON response, potentially exposing sensitive database credentials, internal paths, or environment variables.
**Learning:** Even internal-facing health checks must fail securely because error messages can be logged by infrastructure or accessed by unprivileged monitors. Direct exposure of exception strings is a common source of information leakage.
**Prevention:** Always use generic, safe error messages in API responses (e.g., "Health check failed") and rely on backend server logs to capture the actual exception traces for debugging.
## 2025-12-12 - Prevent Command Injection Bypass in ShellTool
**Vulnerability:** ShellTool's `_is_command_allowed` verified tokens using `if token in dangerous:`, missing bypasses via absolute paths (`/bin/rm`), relative paths (`./rm`), or assignment flags (`--exec=/bin/rm`).
**Learning:** Checking for command injection requires strict validation of the basename of executable targets, not just exact token matches, as interpreters and shell wrappers process commands differently.
**Prevention:** Use `pathlib.Path(token).name` for command token validations to extract the base command, and correctly parse options with assignments to prevent nested bypasses.
## 2025-12-11 - Prevent Denial of Service (DoS) via Unclosed SQLite Connections
**Vulnerability:** Found a resource leak vulnerability in `src/data_processing/data_processor/python/data_processor/file_utils.py` where a `sqlite3` connection was established but not guaranteed to close if an exception occurred during data processing (`pd.read_sql_query` or `data.to_sql`).
**Learning:** `sqlite3.connect()` creates a context manager that manages *transactions* (commit/rollback), not the connection lifecycle itself. It will not close the connection when exiting the `with` block. Failing to close connections explicitly can exhaust system file descriptors, leading to Denial of Service (DoS), or leave the database in a locked state.
**Prevention:** Always use `contextlib.closing` to wrap `sqlite3.connect()` (i.e. `with closing(sqlite3.connect(...)) as conn:`) to guarantee the connection is closed even if an exception occurs during the database operation.
## 2025-02-24 - Improve CLI tools blocklist validation
**Vulnerability:** Found a validation bypass in `cli_tools.py` where a user-controlled token was directly compared with a list of dangerous executables without properly handling whitespace characters around it. This could allow dangerous shell commands wrapped with spaces (e.g., `--exec="  /bin/rm  "`) to bypass the `token in dangerous` and assignment parsing checks.
**Learning:** Checking for blocklisted items must thoroughly normalize the input strings (e.g. using `strip()`) before performing string and path-matching comparisons to prevent bypass by padding with spaces.
**Prevention:** Always normalize the strings extracted from user inputs using `.strip()` before verifying them against a blocklist, especially when processing tokenized arguments parsed from shell commands.
## 2026-06-16 - [Missing auth on power_supply_integration.py routers]
**Vulnerability:** Found unauthenticated API routes in `power_supply_integration.py` that could modify configuration and setpoints.
**Learning:** Newly created routers (like `power_supply`) aren't automatically protected by the main app's dependencies.
**Prevention:** Apply `Depends(require_admin_key)` to mutating endpoints inside newly added APIRouters.
## 2026-06-29 - Prevent Denial of Service (DoS) via Unclosed SQLite Connections in event_logger.py
**Vulnerability:** Found a resource leak vulnerability in `src/p1am_control_system/desktop/event_logger.py` where a `sqlite3` connection was established but not guaranteed to close if an exception occurred during database operations because `sqlite3.connect()` context manager only manages transactions, not the connection lifecycle.
**Learning:** Failing to close connections explicitly can exhaust system file descriptors, leading to Denial of Service (DoS), or leave the database in a locked state.
**Prevention:** Wrapped `sqlite3.connect()` with `contextlib.closing()` to guarantee the connection is closed even if an exception occurs during the database operation.

## 2024-07-06 - Missing AST Security Gate on ODE Solver
**Vulnerability:** Code injection vulnerability found in `TI89Calculator._solve_differential_equation_cached` because it bypassed `_ast_security_gate` before passing input to `parse_expr` which runs `eval`.
**Learning:** All paths taking untrusted math equations directly to SymPy parsers need structural validation.
**Prevention:** Ensure all evaluation points explicitly invoke `_ast_security_gate`.
## 2025-02-27 - Fail-Closed Security Boundary for Evaluated Untrusted Expressions
**Vulnerability:** SymPy's `parse_expr` uses `eval()` and requires upstream validation. The `_ast_security_gate` structural validator was fail-open when encountering a `SyntaxError` while using `ast.parse(stripped, mode="eval")`, relying on `parse_expr` as a backstop. This could allow non-standard Python syntax (e.g. `x = y` or sympy specific forms) to bypass the security gate entirely.
**Learning:** Security validation gates designed to protect `eval`-like functions must be fail-closed. If structural validation fails or raises an error, the input must be explicitly rejected rather than implicitly passed to a dangerous downstream execution context.
**Prevention:** Catch parsing exceptions (like `SyntaxError` in AST gates) and explicitly raise a rejection error (e.g., `ValueError`) to ensure the security gate strictly enforces an allowlist.
## 2026-08-24 - [Mitigate XSS in KaTeX Derivation component]
**Vulnerability:** KaTeX formula output was passed directly into `dangerouslySetInnerHTML` without sanitization.
**Learning:** Even math/formula rendering libraries can sometimes be tricked into producing malicious HTML output. It should always be sanitized before being injected into the DOM.
**Prevention:** Use `DOMPurify.sanitize` to wrap the generated HTML strings before injection.
