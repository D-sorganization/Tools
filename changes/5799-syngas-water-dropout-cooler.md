---
issue: 5799
summary: "Evaluate water knockout at cooler temperature and re-normalize composition (TOOLS-05)"
branch: "fix/5799-syngas-water-dropout-cooler"
dl_state: "in_review"
next_step: "Merge the PR."
---

Water knockout in syngas multistage compression is evaluated at cooler outlet temperature (default 313.15 K) instead of superheated compressor discharge temperature, gas composition is re-normalized after each stage, and postconditions (dropout >= 0 and sum(comp) == 1) are enforced.
