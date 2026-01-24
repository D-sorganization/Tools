# Assessment: Security (Category F)

## Grade: 7/10

## Evidence
- **Path Traversal Protection**: `UnifiedToolsLauncher.py` explicitly validates paths against `..` and ensures they are within the repo root.
- **Input Sanitization**: `TI89Calculator` sanitizes inputs and restricts globals (`_SAFE_GLOBALS_CACHE`) to prevent code execution vulnerabilities.
- **Dependency Scanning**: The CI/CD pipeline runs `pip-audit`, although it is currently configured to not fail on error (`|| true`), which weakens its enforcement.
- **Secret Management**: `AGENTS.md` strictly forbids committing secrets.
- **Permissions Policy**: The calculator web app explicitly sets `Permissions-Policy` headers to disable sensitive features (camera, mic, etc.).

## Recommendations
1. **Enforce Audit**: Configure `pip-audit` to fail the build on critical vulnerabilities (remove `|| true` in CI for high severity).
2. **Sanitize HTML**: Ensure all user-supplied data in `UnifiedToolsLauncher.py` (tool names/descriptions) is HTML-escaped before rendering in Qt widgets.
3. **AST Safety**: Verify `Data_Processor_r0.py` custom variable formula evaluation uses `ast.literal_eval` or a restricted environment, not `eval()`.
