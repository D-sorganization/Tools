---
issue: 5847
summary: "Do not divide operating_current_a by electrode count in electrode advisor layout (TOOLS-09)"
branch: "fix/5847-electrode-operating-current"
dl_state: "in_review"
next_step: "Merge PR"
---

In 3-phase AC electric arc furnaces (EAF) or submerged arc furnaces (SAF) with 3 electrodes, each electrode is connected to an individual electrical phase and carries the full line operating current. Correct default electrode placements in Python (`_build_default_placements`) and Rust (`build_default_placements`) to assign `operating_current_a` directly without dividing by electrode count (TOOLS-09, #5847).
