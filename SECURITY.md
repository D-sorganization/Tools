# Security Policy

## Supported Versions

Security fixes are applied to the latest code on `main`.
Older branches and unpublished snapshots are handled on a best-effort basis.

## Reporting a Vulnerability

If you discover a security issue in this repository:

1. Do not open a public GitHub issue.
2. Use GitHub's private vulnerability reporting flow for this repository if it is enabled.
3. If private reporting is unavailable, contact the maintainers directly before any public disclosure.

Please include:

- a clear description of the issue
- affected paths or components
- reproduction steps or a proof of concept
- the likely impact
- any mitigation ideas you already have

## Response Expectations

- Acknowledgment within 2 business days
- Initial triage within 7 days
- Coordinated disclosure after a fix or mitigation is available

## Scope

This policy covers:

- shared Python packages and launchers
- build, release, and CI automation
- repository tooling that ships to or supports downstream repos

## False Positives

Some test fixtures intentionally use token-shaped strings to exercise parsing and header handling. In particular, the CSRF tests under `src/media_processing/video_processor/apps/web/lib/__tests__/` and the API header tests under `tests/` may contain synthetic values that are not credentials and should not be treated as secrets during scanning.
