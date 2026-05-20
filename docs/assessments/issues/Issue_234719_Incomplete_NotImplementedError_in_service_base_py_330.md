---
title: "Incomplete NotImplementedError in service_base.py:330"
labels: ["incomplete-implementation", "critical", "high-impact"]
assignee: "unassigned"
status: "open"
---

# Issue Description

Found critical incomplete implementation in `./src/shared/python/chat/service_base.py` at line 330.

## Context

**Type**: NotImplementedError | **Location**: `./src/shared/python/chat/service_base.py:330`

```python
raise NotImplementedError("refresh_models must be implemented by subclass")
```

## Audit Metrics

- **Impact**: 5/5 | **Coverage**: 3/5 | **Complexity**: 4/5

## Recommendation

Implement missing logic or document the rationale for the gap.
