---
issue: 5836
summary: "Correct KG_TO_MG constant from 1000.0 to 1e6 in scrubber calculator (TOOLS-22)"
branch: "fix/5836-scrubber-kg-to-mg-constant"
dl_state: "in_review"
next_step: "Merge the PR."
---

In `sidekick.process_calculators.scrubber_calculator`, the constant `KG_TO_MG` was previously set to `1_000.0` (which represented grams per kilogram). Corrected `KG_TO_MG` to `1_000_000.0` ($1\text{ kg} = 10^6\text{ mg}$) in alignment with `KG_TO_MG_PER_L = 1_000_000.0`. Added dedicated verification-first unit test in `test_scrubber_calculator.py`.
