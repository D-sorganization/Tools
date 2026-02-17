---
title: "Security: Remove sensitive .msg files from repository"
labels:
  ["jules:assessment,needs-attention", "security", "urgent", "data-leakage"]
assignees: []
---

## Description

**Assessment Found**: Category F (Security) - Grade 4/10

The repository contains binary Outlook `.msg` files in `src/shared/python/upstream_drift_tools/process_calculators/psa_package/References/Email Correspondence/`.
These files contain actual email correspondence and likely include PII (Personally Identifiable Information) or proprietary internal communications that should not be in a source code repository.

## Impact

- **High**: Potential leak of sensitive data.
- Bloats repository size.

## Action Items

1. [ ] Confirm with stakeholders that these files can be deleted.
2. [ ] Remove files from the current `HEAD`.
3. [ ] (Optional but recommended) Rewrite git history to remove them completely using `git filter-repo` or BFG.
4. [ ] Verify `*.msg` is in `.gitignore` (Already done by Assessment Generator).

## Related Files

- `src/shared/python/upstream_drift_tools/process_calculators/psa_package/References/Email Correspondence/*.msg`
