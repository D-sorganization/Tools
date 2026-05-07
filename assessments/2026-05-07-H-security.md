# Criterion H: Security

**Repo:** Tools
**Score:** 70/100
**Weight:** 10%
**Weighted Contribution:** 7.00

## Evidence

```json
{
  "secrets_raw": 5,
  "bandit_cfg": 0,
  "security_md": 1
}
```

## Findings

### P1: [Tools] 5 potential hardcoded secrets detected

Audit source files for hardcoded credentials. Move to environment variables or secret manager (Vault, AWS Secrets Manager).
