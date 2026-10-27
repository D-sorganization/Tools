---
issue: 5823
summary: "Include entropy of mixing in ideal gas mixture entropy calculation (TOOLS-07)"
dl_state: "in_review"
branch: "fix/5823-thermo-mixture-entropy-of-mixing"
next_step: "Merge the PR."
---

Include ideal gas entropy of mixing Delta S_mix = -R sum(x_i ln x_i) in mixture
molar entropy evaluation to restore thermodynamic consistency.
