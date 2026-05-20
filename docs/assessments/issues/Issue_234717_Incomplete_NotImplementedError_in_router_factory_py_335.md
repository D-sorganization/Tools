---
title: "Incomplete NotImplementedError in router_factory.py:335"
labels: ["incomplete-implementation", "critical", "high-impact"]
assignee: "unassigned"
status: "open"
---

# Issue Description

Found critical incomplete implementation in `./src/shared/python/chat/router_factory.py` at line 335.

## Context

**Type**: NotImplementedError | **Location**: `./src/shared/python/chat/router_factory.py:335`

```python
except NotImplementedError as exc:
```

## Audit Metrics

- **Impact**: 5/5 | **Coverage**: 3/5 | **Complexity**: 4/5

## Recommendation

Implement missing logic or document the rationale for the gap.
