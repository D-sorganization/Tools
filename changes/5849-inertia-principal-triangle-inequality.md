---
issue: 5849
summary: "Validate triangle inequality on principal moments of inertia (TOOLS-17)"
branch: "fix/5849-inertia-principal-triangle-inequality"
dl_state: "in_review"
next_step: "Merge PR"
---

Physical inertia tensors must satisfy the triangle inequality strictly on their principal moments of inertia ($\lambda_3 \le \lambda_1 + \lambda_2$ where $\lambda_i$ are the eigenvalues of the inertia tensor). Coordinate-frame diagonal components frequently pass diagonal checks even when the eigenvalues violate the triangle inequality (>26% false acceptance rate). Update `_validate_inertia` in `src/inertia_calculator/python/inertia_calculator/ui/pyqt6/main_window.py` to evaluate eigenvalues using `np.linalg.eigvalsh` and assert the triangle inequality on principal moments (TOOLS-17, #5849).
