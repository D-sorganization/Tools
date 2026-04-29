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

---

## Secrets & Credentials Management (Issue #2356)

This section addresses minimizing secret-keyword exposure and preventing hardcoded secrets in the repository.

### Critical Rules

**NEVER commit to the repository:**
- API keys (OpenAI, Google, AWS, GitHub, Stripe, etc.)
- Database credentials or connection strings with passwords
- Private encryption keys or SSH keys
- OAuth tokens or bearer tokens
- AWS/Azure/GCP service account credentials
- Any other sensitive credentials or secrets

###  Best Practices

#### 1. Use Environment Variables

Load all secrets at runtime from environment variables:

```python
import os

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")
```

With `python-dotenv` for local development:

```python
from dotenv import load_dotenv
import os

load_dotenv()  # Loads from .env (excluded from version control)
api_key = os.getenv('API_KEY')
```

#### 2. Create `.env.example` Templates

For each component requiring secrets, maintain a `.env.example` showing the expected structure:

```bash
# .env.example - DO NOT include real values
GEMINI_API_KEY=your_key_here
GITHUB_TOKEN=your_token_here
DATABASE_URL=postgresql://user:password@localhost/dbname
```

Examples in this repo:
- `.env.example` — Flask/calculator app secrets
- `src/document_processing/pdf_renamer/.env.example` — Gemini API key

#### 3. Use OS Keyring for Interactive Tools

For GUI applications that persist user credentials, use `python-keyring` to store in OS-native secure storage:

```python
import keyring

# Save to OS keyring
keyring.set_password("app_name", "username", secret_value)

# Retrieve from OS keyring
secret_value = keyring.get_password("app_name", "username")
```

Example: `src/document_processing/pdf_renamer/config.py` demonstrates this pattern.

#### 4. Exclude Config Files

The `.gitignore` excludes:
- `.env*` files (all environment variable files)
- `*.key`, `*.pem` (certificates and keys)
- `secrets/` directories
- `config/secrets/` directories
- Cloud provider credential files

When adding new config files containing secrets, update `.gitignore`.

#### 5. Scan Before Committing

Use the built-in secrets scanner:

```bash
python3 -m src.python.src.utils.secrets_scanner src/
```

This detects patterns matching:
- AWS keys: `AKIA` prefix
- GitHub tokens: `ghp_` prefix
- OpenAI keys: `sk-` prefix
- Slack tokens: `xox*` prefix
- Private keys: `-----BEGIN PRIVATE KEY-----`

#### 6. Code Review Checklist

During code review, verify:

1. **String literals**: No hardcoded API keys, passwords, or tokens
2. **Docstrings & Examples**: No real credentials in documentation
3. **Test Fixtures**: Use placeholder values only
4. **Logging**: Never log secrets
5. **Exceptions**: Don't expose secrets in error messages

###  Examples from This Repository

**PDF Renamer** (`src/document_processing/pdf_renamer/`):
- Demonstrates priority-ordered secret retrieval: environment → OS keyring → none
- Uses interactive setup to store credentials securely
- See `config.py` for the pattern

**Folder Packer Pro** (`src/folder_packer_pro/`):
- User-provided passwords (never hardcoded)
- Uses PBKDF2 + AES-256 for encryption
- Passwords stay in memory only during operations

### If a Secret Was Accidentally Committed

1. **Notify the team immediately** via email (not a public issue)
2. **Revoke the credential** (regenerate API keys, rotate passwords)
3. **Remove from history** with team approval (force-push if needed)
4. **Monitor for abuse** of the exposed credential

### References

- [OWASP: Secrets Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [CWE-798: Use of Hard-Coded Credentials](https://cwe.mitre.org/data/definitions/798.html)
- [python-keyring Documentation](https://keyring.readthedocs.io/)
- [python-dotenv Documentation](https://python-dotenv.readthedocs.io/)
