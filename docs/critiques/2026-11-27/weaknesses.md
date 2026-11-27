# Weaknesses — Tools — 2026-11-27

Reviewed at `main` HEAD `a8c95b731`. Scope: the calculator Agent API (epic
[#6019](https://github.com/D-sorganization/Tools/issues/6019)), which in
November 2026 began serving 18 `calc_backend` calculators to LLM agents over
stdio MCP and REST with a machine-readable provenance block. Agents act on
that envelope without a human reading the design manual, so whatever it
asserts gets read as fact.

## Weakness: `calculation_status: "verified-unapproved"` stamped on calculations with no formula linkage

### Summary of Concern

Every MCP response reports `provenance.calculation_status ==
"verified-unapproved"`, but no calculator is linked to the design manual.
The status therefore asserts a verification that the repository's own
governance says has not happened.

### Location

- **File/Document:** `src/shared/python/calc_backend/agent_api.py`;
  `manuals/tools/agent_api_calculator_bindings.json`;
  `src/shared/python/calc_backend/registry/catalog.py`
- **Section:** `DEFAULT_CALCULATION_STATUS` (line 19); `calculation_status()`
  (lines 100–111); `build_provenance()` (lines 147–162)
- **Claim or Equation:** AGENTS.md § Result conventions: responses carry a
  "conservative `calculation_status` (design-manual approval is never
  implied)". AGENTS.md § Engineering Design Manual Authority: "An exemplar
  may claim `verified-unapproved` only when its calculation, chapter, module,
  source, consumer, test, and golden-fixture links all resolve."

### Nature of the Issue

- [x] Logical gap (missing evidence falls back to the second-strongest
      status in the vocabulary instead of the weakest one)
- [ ] Unstated assumption
- [x] Overgeneralization (one status applied to 18 heterogeneous calculators)
- [ ] Empirical insufficiency
- [x] Terminological ambiguity (`verified-unapproved` means different things
      in the manual and in MCP envelopes)
- [ ] Literature conflict

### Why This Is a Problem

`agent_api_calculator_bindings.json` ships `"bindings": {}`, and every
`CATALOG` entry has `"formula_ids": ()`. `calculation_status(())` returns
`DEFAULT_CALCULATION_STATUS`, which is `"verified-unapproved"`. An unresolved
formula ID also falls back to that value (line 107). So does a request for an
unknown calculator id, because `build_provenance()` is called with no
arguments (`mcp_server.py:93`). In practice the label means "nothing was
looked up", but it reads as "verified, awaiting approval".

The manual uses the same term as a strict gate. In
`manuals/tools/schemas/exemplar-coverage.schema.json` it requires every link
in the chain to resolve. The two meanings now share one string. A downstream
agent, or a reviewer reading a log, cannot tell the swing D-plane exemplar
(actually traced) apart from `syngas-compression`. The October critique
(§3) found that calculator's constant-γ chain unvalidated, yet it returns
the same status.

Tests lock in the overclaim instead of catching it:
`tests/calc_backend/test_syngas_compression_api.py:110` and
`tests/calc_backend/test_water_vapor_pressure_api.py:51` assert
`== "verified-unapproved"` for calculators with empty `formula_ids`.

Secondary: `tools_version()` falls back to `"0.0.0"` and `git_sha()` to
`"unknown"` (`agent_api.py:35,56`). In an installed wheel with no `.git` and
no `TOOLS_GIT_SHA`, the provenance block identifies neither the code nor the
release. Nothing in the envelope marks this as degraded.

### Evidence / References

- `manuals/tools/agent_api_calculator_bindings.json` — `"bindings": {}`.
- `src/shared/python/calc_backend/registry/catalog.py` — 18 entries, all
  `"formula_ids": ()`.
- `src/shared/python/calc_backend/agent_api.py:19,100–111`.
- `manuals/tools/schemas/exemplar-coverage.schema.json:77,101` (strict
  meaning of the same status).

### Severity

- [ ] Low
- [ ] Medium
- [x] High (provenance is the core claim of the Agent API; it currently
      claims more than the evidence supports for every calculator)

### Suggested Remedies

- Add a distinct status for "no linkage", such as `unlinked` or
  `provisional` (the manual vocabulary already has `provisional`), and make
  it the default for empty or unresolved `formula_ids` and for
  unknown-calculator envelopes.
- Change the two tests above so they assert the weakest status when
  `formula_ids == ()`. Add a contract test that `verified-unapproved` is
  returned only when every formula ID resolves in `calculation-registry.json`.
- Populate `agent_api_calculator_bindings.json` before claiming epic #6019
  acceptance, or state in `llms.txt` that no calculator is linked yet.
- Report a degraded-provenance warning when `git_sha == "unknown"` or
  `tools_version == "0.0.0"`.

---

## Weakness: `maturity: "stable"` is undefined and contradicts open evidence gaps

### Summary of Concern

`list_calculators` advertises 13 of 18 calculators as `"stable"` and 5 as
`"beta"`. The term is defined nowhere: not in the spec, AGENTS.md,
`llms.txt`, `API_STANDARDIZATION.md`, or the conformance template. Several
"stable" entries have known open validity gaps or recent correctness fixes.

### Location

- **File/Document:** `src/shared/python/calc_backend/registry/spec.py`
  (`Maturity = str`); `src/shared/python/calc_backend/registry/catalog.py`;
  `src/shared/python/calc_backend/mcp_server.py:51`
- **Section:** `CATALOG[*]["maturity"]`
- **Claim or Equation:** `"maturity": "stable"` for `syngas-water`,
  `acid-gas-dewpoint`, `wgs-reactor`, `ode-solver`, and nine others.

### Nature of the Issue

- [ ] Logical gap
- [x] Unstated assumption (what "stable" requires is never written down)
- [x] Overgeneralization
- [x] Empirical insufficiency
- [x] Terminological ambiguity
- [ ] Literature conflict

### Why This Is a Problem

An agent choosing between calculators will reasonably read `stable` as
"validated, safe to rely on". The labels do not follow any evidence ordering
a reviewer could reconstruct:

- **`syngas-water` is `stable`.** Its worked example runs at 30 bar, which
  is the high-pressure regime where the 2026-10-30 critique §2 found the
  ideal-gas saturation / relative-humidity heuristic unvalidated. Its g/m³
  and lb/MMscf conversions were wrong until
  [#6007](https://github.com/D-sorganization/Tools/pull/6007) (2026-11-12).
- **`water-vapor-pressure` is `beta`.** It is the IAPWS saturation primitive
  that a water-content calculation rests on. The derived calculation is
  labelled more mature than its input.
- **`acid-gas-dewpoint` is `stable`.** The 2026-10-30 critique §1 found that
  its `validity_limits` are never enforced and that HF/H₂S still use
  pure-component Antoine inversion in wet gas.
- **The `ode-solver` worked example contradicts its handler.** The catalog
  records `response_summary={"success": False}` (`catalog.py:176`), but
  `_rk4_solve` either returns `success=True` or raises
  (`routers/ode_solver.py:113–144`). The documented outcome of the canonical
  example therefore cannot occur. A "stable" calculator carries a worked
  example nobody has executed.

Only one `response_summary` out of 18 (`financial`) is checked against a real
response (`tests/calc_backend/test_calculator_registry_contract.py:126`). The
other 17 are unverified prose that is served to agents as documentation.

### Evidence / References

- `src/shared/python/calc_backend/registry/spec.py:12` — `Maturity = str`.
- Repo-wide search for `maturity` in `docs/`, `llms.txt`, `AGENTS.md`,
  `calc_backend/*.md`, `config/fleet_agent_api_conformance/` finds no
  definition.
- Fleet Critic 2026-10-30, §1–§3 (`docs/critiques/2026-10-30/weaknesses.md`).

### Severity

- [ ] Low
- [x] Medium (no calculation is wrong because of the label itself, but it
      tells agents the opposite of what the evidence says)
- [ ] High

### Suggested Remedies

- Turn `Maturity` into a `Literal`/`Enum` with written entry criteria. For
  example, `stable` could require linked `formula_ids`, golden-value tests,
  an enforced validity envelope, and no open critique findings of Medium
  severity or higher.
- Until those criteria exist, downgrade every calculator to `beta` or
  `experimental`.
- Execute every `worked_example` against its handler in CI, and assert
  `response_summary` the same way the `financial` test does. Fix the
  `ode-solver` summary.

---

## Weakness: The MCP `units` field describes inputs, not the returned outputs

### Summary of Concern

`run_calculation` returns `{"outputs": ..., "units": ...}` side by side, but
`units` is a copy of the catalog's request-field units. The numbers in
`outputs` have no unit metadata at all.

### Location

- **File/Document:** `src/shared/python/calc_backend/mcp_server.py:162–169`
  (`units=dict(spec.units)`); `src/shared/python/calc_backend/registry/catalog.py`
- **Section:** `build_result(..., outputs=outputs, units=dict(spec.units))`
- **Claim or Equation:** Fleet Agent API envelope `units` adjacent to
  `outputs`.

### Nature of the Issue

- [ ] Logical gap
- [x] Unstated assumption (that suffixed field names are enough for unit
      safety)
- [ ] Overgeneralization
- [ ] Empirical insufficiency
- [x] Terminological ambiguity
- [ ] Literature conflict

### Why This Is a Problem

For `syngas-compression`, the envelope `units` maps `inlet_pressure_bar`,
`flow_rate_kmol_h` and similar request fields, while `outputs` contains
`power_hp`, `outlet_temperature_k` and `work_actual_j_per_mol`. The test
confirms the input-unit behaviour
(`tests/calc_backend/test_syngas_compression_api.py:109`). For
`water-vapor-pressure`, `units` is `{"temperature_c": "degC"}` and the answer
is `pressure_pa`. For `flow-rate`, `units` is `{"value": "varies"}`.

An agent that reads the envelope structurally, which is what it exists for,
has no way to get the unit of the answer from the field labelled `units`. It
has to parse suffixes from output names, and some outputs have none (for
example `heat_capacity_ratio`, or `results.total_revenue` in `financial`).
This is the gap that a unit-safe interchange envelope is supposed to close.
The repo has just fixed one unit-conversion defect (#6007), which shows
these errors happen.

### Evidence / References

- `src/shared/python/calc_backend/mcp_server.py:167`.
- `tests/calc_backend/test_syngas_compression_api.py:105–110`.
- `catalog.py` `water-vapor-pressure` and `flow-rate` entries.

### Severity

- [ ] Low
- [x] Medium (a misread unit propagates silently into downstream agent
      reasoning)
- [ ] High

### Suggested Remedies

- Split the envelope into `input_units` and `output_units`, or derive
  `output_units` from the response model's field metadata.
- Add a contract test that every numeric leaf in `outputs` has a declared
  unit (or an explicit `"dimensionless"`).
- Document the field's meaning in `llms.txt` and the conformance template.
