# Criterion D: Error Handling

**Repo:** Tools
**Score:** 13.100000000000001/100
**Weight:** 10%
**Weighted Contribution:** 1.31

## Evidence

```json
{
  "bare_except": 5,
  "except_exception": 50,
  "noqa_suppressions": 369
}
```

## Findings

### P1: [Tools] 5 bare `except:` statements

Replace bare `except:` with specific exception types. Follow 'Crash Early' principle.

### P1: [Tools] 369 lint/type suppressions

High suppression count indicates over-suppression or real code quality issues. Audit and fix root causes.
