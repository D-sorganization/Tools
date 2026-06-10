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
