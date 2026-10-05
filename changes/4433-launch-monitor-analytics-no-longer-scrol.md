---
issue: 4433
summary: "Launch Monitor Analytics no longer scrolls sideways at 390x844 once results render: the analysis grid uses minmax(0,1fr) tracks with min-w-0 children so result tables scroll inside their own overflow-x-auto cards; the every-state Playwright pass now asserts zero document overflow."
dl_state: "in_review"
next_step: "Merge the PR; the 1440x900 evidence captures are layout-identical and need no refresh."
branch: "fix/lma-mobile-overflow"
---
