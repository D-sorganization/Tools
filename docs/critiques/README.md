# Critiques

Adversarial scientific review of this repository, run by the Fleet Critic role
(`Repository_Management/docs/fleet-critic.md`) on a quarterly cadence for
`Tools`. The Critic assumes the author is competent, assumes external
reviewers are hostile and rushed, and surfaces weaknesses in documented
claims, model assumptions, and validation evidence before they do.

The Critic finds holes; it does not patch them. Remediation is tracked
through normal issues/PRs, linked from each report below.

## Methodology

Each pass reviews, per the playbook:

- All documentation added or materially changed in the last 30 days
  (`docs/**`).
- New or modified docstrings carrying scientific/algorithmic claims
  (`src/**`).
- Scientific or algorithmic claims in `README.md` / `SPEC.md`.

Findings are classified by nature (logical gap, unstated assumption,
overgeneralization, empirical insufficiency, terminological ambiguity,
literature conflict) and severity (Low / Medium / High), per
`weaknesses.md` in each dated subdirectory.

## Reports

| Date       | Summary                                                                                                                                                                                                                |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-10-02 | [summary](2026-10-02/summary.md) — a merged automated PR silently reverted a security fix and a published scientific correction back onto `main`; two secondary evidentiary gaps in the impact-acoustics program docs. |
| 2026-10-30 | [summary](2026-10-30/summary.md) — after October board fixes for aqueous HCl dew point and intercooler knockout temperature, three medium evidence gaps remain in Kiang applicability, high-P water dropout modeling, and constant-γ compression ([#5870](https://github.com/D-sorganization/Tools/issues/5870)). |
