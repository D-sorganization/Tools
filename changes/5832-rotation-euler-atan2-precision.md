---
issue: 5832
summary: "Use atan2 and hypot for Euler angle extraction near singularity (TOOLS-28)"
branch: "fix/5832-rotation-euler-atan2-precision"
dl_state: "in_review"
next_step: "Merge the PR."
---

In `rotation_converter.core._rotation_matrix_to_euler_impl`, middle angle extraction previously used `math.acos(cb)` for proper Euler conventions and `math.asin(sb)` for Tait-Bryan conventions, yielding round-trip precision loss near gimbal lock due to infinite derivative of $\arcsin$/$\arccos$ at $\pm 1$. Replaced with `math.atan2(math.hypot(R[j, i], R[other, i]), cb)` for proper Euler and `math.atan2(sb, math.hypot(R[i, i], R[i, j]))` for Tait-Bryan conventions, and tightened degenerate gimbal threshold to $10^{-12}$. Added precision tests in `test_rotation_core.py`.
