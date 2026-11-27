# Fleet Critic — Tools — 2026-11-27

## Executive Summary

In November 2026 the main new claim surface in Tools was the calculator
Agent API (epic [#6019](https://github.com/D-sorganization/Tools/issues/6019)).
It serves 18 `calc_backend` calculators to LLM agents over MCP/REST, with a
provenance block and maturity labels. This pass looked at what that envelope
asserts, and found that it asserts more than the evidence behind it.

1. **High — every MCP result is stamped `calculation_status:
   "verified-unapproved"` although no calculator is linked to the design
   manual.** `agent_api_calculator_bindings.json` is empty and every catalog
   `formula_ids` is `()`. The fallback for "nothing resolved" is the same
   string that AGENTS.md reserves for fully traced exemplars, and two tests
   assert it. See `weaknesses.md` §1.
2. **Medium — `maturity: "stable"` is undefined, and is applied to
   calculators with open evidence gaps.** Examples are `syngas-water`
   (critique 2026-10-30 §2; unit bug fixed in #6007 on 2026-11-12) and
   `acid-gas-dewpoint` (2026-10-30 §1). The `ode-solver` worked example
   documents `success: False`, which its handler cannot return. Only 1 of 18
   worked-example summaries is checked against a real response. See §2.
3. **Medium — the envelope's `units` field lists request-input units, not
   the units of the returned `outputs`.** Agents cannot get the unit of the
   answer from the field named `units`. See §3.

## What Would Change My Mind

- §1 would close with a weakest-by-default status (such as `unlinked` or
  `provisional`) for empty or unresolved `formula_ids`, plus a contract test
  that `verified-unapproved` appears only when every formula ID resolves in
  `calculation-registry.json`.
- §2 would close with a closed `Maturity` enum that has written entry
  criteria, and CI that executes every worked example and asserts its
  `response_summary`.
- §3 would close with `output_units` derived from response models and a test
  that every numeric output leaf carries a unit.

## Strongest Version of the Author's Argument

The Agent API is explicitly conservative in one direction: it never emits
`approved` without a registry approval record (`agent_api.py:77–80`), and
FAILED envelopes carry provenance (#6053). The defects above are about the
fallbacks and labels, not the arithmetic. The fixes are small, local, and
mostly need no change to calculator code.

## Prior Findings

The 2026-10-30 remediation tracker
[#5870](https://github.com/D-sorganization/Tools/issues/5870) was closed on
2026-11-01 after the report was documented. The underlying evidence gaps
(Kiang envelope, high-pressure knockout, constant-γ compression) remain
relevant to §2 above, because those calculators are now exposed to agents.

## Scope

Reviewed at HEAD `a8c95b731`:

- `src/shared/python/calc_backend/{agent_api,mcp_server}.py`,
  `registry/{catalog,spec,store,formula_linkage}.py`,
  `routers/ode_solver.py`
- `manuals/tools/agent_api_calculator_bindings.json`
- `tests/calc_backend/` (registry contract, syngas compression, water vapor
  pressure, MCP provenance tests)
- AGENTS.md § Agent API and § Engineering Design Manual Authority
- Commits since 2026-10-28 (216 commits; calculator Agent API, statistics and
  unit-conversion fixes)

Not reviewed in depth this pass: the November `data_processor` statistics
fixes (BCa, t-quantiles, Cook's distance) and the `ai_backend` wheel claims.
They are candidates for the next pass.
