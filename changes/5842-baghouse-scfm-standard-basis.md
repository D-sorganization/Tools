---
issue: 5842
summary: "Support configurable standard temperature reference basis for SCFM calculation"
branch: "fix/5842-baghouse-scfm-standard-basis"
dl_state: "in_review"
next_step: "Merge the PR into main."
---

Support standard volumetric flow (SCFM) reference basis configuration (`standard_basis`: `"STP"`, `"EPA"`/`"NTP"`, `"STANDARD_60F"`/`"SATP_60F"`) in `BaghouseCalculator` to align with US EPA 40 CFR § 60.2 (20 °C) and industrial filtration vendor specifications (60 °F).
