# Assessment N: Visualization & Export

## Executive Summary

- Conducted specific targeted review for Visualization & Export.
- Findings: Config is scattered between .env, json, and hardcoded values. Missing defaults.
- The prompt guidelines were applied specifically to this category.
- Critical gaps identified requiring immediate attention.

## Top 10 Risks

1. **Major - Hardcoded configuration values.**
2. **Major - Secrets committed to config files.**
3. **Minor - Missing config schemas.**
4. **Minor - Env vars not documented.**
5. **Minor - Difficult to override configs for tests.**

## Scorecard

| Metric | Score | Evidence |
|---|---|---|
| Core Implementation | 6.5/10 | Config is scattered between .env, json, and hardcoded values. Missing defaults. |
